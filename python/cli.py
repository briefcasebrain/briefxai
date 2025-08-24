#!/usr/bin/env python3
"""
BriefX CLI - Command line interface for BriefX conversation analysis
"""

import asyncio
import json
import logging
import webbrowser
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.panel import Panel

# Import our modules
from briefx.data.parsers import parse_json_conversations
from briefx.examples import generate_example_conversations

console = Console()

@click.group(invoke_without_command=True)
@click.option('--port', '-p', default=8080, help='Port for web interface')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--version', is_flag=True, help='Show version')
@click.pass_context
def cli(ctx, port, verbose, config, version):
    """BriefX - Advanced AI conversation analysis platform"""
    
    if version:
        console.print("BriefX v2.0.0 (Python Edition)")
        return
    
    # If no subcommand provided, default to ui
    if ctx.invoked_subcommand is None:
        ctx.invoke(ui)
    
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['port'] = port
    ctx.obj['verbose'] = verbose
    ctx.obj['config_path'] = config
    
    # Setup logging
    setup_logging(verbose=verbose)
    
    # Load configuration
    briefx_config = BriefXConfig()
    if config:
        # Load config from file (JSON format)
        try:
            with open(config, 'r') as f:
                config_data = json.load(f)
                briefx_config = BriefXConfig.from_dict(config_data)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return
    
    briefx_config.website_port = port
    briefx_config.verbose = verbose
    
    ctx.obj['config'] = briefx_config
    
    # Initialize analysis pipeline
    initialize_pipeline(briefx_config)

@cli.command()
@click.option('--port', '-p', type=int, help='Port to serve on')
@click.option('--open', 'open_browser', is_flag=True, default=True, help='Automatically open browser')
@click.pass_context
def ui(ctx, port, open_browser):
    """Start the interactive web UI (default)"""
    
    config = ctx.obj['config']
    if port:
        config.website_port = port
    
    console.print(Panel.fit(
        f"[bold green]🚀 Starting BriefX Web Interface[/bold green]\n\n"
        f"[bold]URL:[/bold] http://localhost:{config.website_port}\n"
        f"[bold]Features:[/bold]\n"
        f"  • Upload conversations or use example data\n"
        f"  • Configure analysis settings\n"
        f"  • View real-time results\n\n"
        f"[dim]Press Ctrl+C to stop the server[/dim]",
        title="BriefX Ready!",
        border_style="green"
    ))
    
    # Start the Flask app
    try:
        # Import here to avoid circular imports
        from app import app
        app.config['PORT'] = config.website_port
        
        # Open browser after short delay if requested
        if open_browser:
            import threading
            import time
            
            def open_browser_delayed():
                time.sleep(1.5)  # Give server time to start
                url = f"http://localhost:{config.website_port}"
                try:
                    webbrowser.open(url)
                    console.print(f"[green]✓[/green] Browser opened to {url}")
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Could not open browser automatically: {e}")
            
            browser_thread = threading.Thread(target=open_browser_delayed)
            browser_thread.daemon = True
            browser_thread.start()
        
        app.run(host='0.0.0.0', port=config.website_port, debug=config.debug)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")

@cli.command()
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True), 
              help='Input data file (JSON)')
@click.option('--output', '-o', 'output_dir', default='output', type=click.Path(), 
              help='Output directory')
@click.pass_context
def analyze(ctx, input_file, output_dir):
    """Analyze from command line (advanced)"""
    
    config = ctx.obj['config']
    
    console.print(f"[bold]Running BriefX Analysis[/bold]")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output_dir}")
    
    # Run analysis
    asyncio.run(run_cli_analysis(config, Path(input_file), Path(output_dir)))

async def run_cli_analysis(config: BriefXConfig, input_path: Path, output_path: Path):
    """Run analysis from CLI with progress display"""
    
    try:
        # Load input data
        console.print(f"\n[bold blue]📁 Loading input data...[/bold blue]")
        with open(input_path, 'r') as f:
            content = f.read()
        
        conversations = parse_json_conversations(content)
        console.print(f"[green]✓[/green] Loaded {len(conversations)} conversations")
        
        # Start analysis with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            analysis_task = progress.add_task("Starting analysis...", total=None)
            
            # Get pipeline and run analysis
            pipeline = get_pipeline()
            session_id = f"cli_{input_path.stem}"
            
            # Monitor progress
            async def update_progress():
                while True:
                    prog = session_manager.get_session_progress(session_id)
                    if prog and 'error' not in prog:
                        status = prog.get('status', 'unknown')
                        if status == 'completed':
                            progress.update(analysis_task, description="✅ Analysis completed!")
                            break
                        elif status == 'failed':
                            progress.update(analysis_task, description="❌ Analysis failed!")
                            break
                        else:
                            msg = prog.get('current_step', 'Processing...')
                            progress_percent = prog.get('progress', 0.0)
                            progress.update(analysis_task, description=f"⚙️ {msg} ({progress_percent:.1f}%)")
                    
                    await asyncio.sleep(0.5)
            
            # Start progress monitoring
            progress_task = asyncio.create_task(update_progress())
            
            try:
                # Run analysis
                results = await pipeline.analyze_conversations(conversations, session_id)
                
                # Cancel progress monitoring
                progress_task.cancel()
                
                # Save results
                console.print(f"\n[bold blue]💾 Saving results...[/bold blue]")
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save analysis results
                results_file = output_path / "analysis_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results.to_dict(), f, indent=2)
                
                # Save completion marker
                marker_file = output_path / "analysis_complete.json"
                marker_data = {
                    "session_id": session_id,
                    "status": "completed",
                    "timestamp": results.conversations[0].to_dict()['timestamp'] if results.conversations else None,
                    "total_conversations": results.total_conversations,
                    "total_clusters": len(results.clusters),
                    "processing_time": results.processing_time
                }
                with open(marker_file, 'w') as f:
                    json.dump(marker_data, f, indent=2)
                
                console.print(f"[green]✓[/green] Results saved to {output_path}")
                console.print(f"[green]✓[/green] Analysis completed in {results.processing_time:.2f}s")
                console.print(f"[dim]Run 'briefx serve -d {output_path}' to view results[/dim]")
                
            except Exception as e:
                progress_task.cancel()
                raise e
                
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        logging.error(f"CLI analysis error: {e}", exc_info=True)

@cli.command()
@click.option('--directory', '-d', 'directory', default='output', type=click.Path(exists=True),
              help='Directory containing results')
@click.pass_context  
def serve(ctx, directory):
    """Serve existing results"""
    
    config = ctx.obj['config']
    
    console.print(f"[bold]Serving existing results from: {directory}[/bold]")
    
    # Check if analysis results exist
    results_path = Path(directory) / "analysis_complete.json"
    if not results_path.exists():
        console.print(f"[yellow]⚠[/yellow] No analysis_complete.json found in {directory}")
        console.print("This directory may not contain BriefX analysis results")
    else:
        # Load and display analysis info
        with open(results_path, 'r') as f:
            analysis_info = json.load(f)
        
        console.print(f"[green]✓[/green] Found analysis results:")
        console.print(f"  Session: {analysis_info.get('session_id', 'unknown')}")
        console.print(f"  Status: {analysis_info.get('status', 'unknown')}")
        console.print(f"  Conversations: {analysis_info.get('total_conversations', 'unknown')}")
        console.print(f"  Clusters: {analysis_info.get('total_clusters', 'unknown')}")
    
    console.print(f"\n[bold green]🌐 Server starting at http://localhost:{config.website_port}[/bold green]")
    
    try:
        # Import and start Flask app
        from app import app
        app.run(host='0.0.0.0', port=config.website_port, debug=config.debug)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")

@cli.command()
@click.option('--output', '-o', 'output_file', default='realistic_conversations.json', 
              type=click.Path(), help='Output file path')
@click.option('--count', '-n', default=50, help='Number of conversations to generate')
@click.pass_context
def example(ctx, output_file, count):
    """Generate realistic example data"""
    
    console.print(f"[bold]Generating {count} realistic example conversations[/bold]")
    
    # Generate example conversations
    conversations = generate_realistic_conversations(count)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump([conv.to_dict() for conv in conversations], f, indent=2)
    
    console.print(f"[green]✓[/green] Generated {len(conversations)} diverse conversations covering:")
    topics = [
        "Technical support", "Creative writing", "Cooking advice", 
        "Health & fitness", "Language learning", "Career advice",
        "Personal finance", "Relationships", "Academic help", "Travel planning"
    ]
    
    for topic in topics:
        console.print(f"  • {topic}")
    
    console.print(f"\n[green]✓[/green] Example data written to {output_file}")

def generate_realistic_conversations(count: int):
    """Generate realistic conversation examples"""
    
    from src.data.models import ConversationData, Message
    import random
    
    # Conversation templates by category
    templates = {
        "Technical Support": [
            ("I'm having trouble logging into my account", "I can help you reset your password. Let me guide you through the process."),
            ("The app keeps crashing when I try to upload files", "That sounds like a file size issue. What type of files are you uploading?"),
            ("Can you explain how to integrate your API?", "Sure! Here's a step-by-step guide to API integration."),
        ],
        "Creative Writing": [
            ("Help me write a story about time travel", "Great idea! Let's start with your protagonist and the time period they visit."),
            ("I need ideas for character development", "Character backstory is key. What's your character's main motivation?"),
            ("How do I write better dialogue?", "Good dialogue reveals character. Try reading it aloud to test authenticity."),
        ],
        "Career Advice": [
            ("Should I negotiate my salary offer?", "Yes, negotiation is normal. Research market rates for your position first."),
            ("How do I ask for a promotion?", "Schedule a meeting with your manager to discuss your career progression."),
            ("I want to change careers but feel stuck", "Career transitions are challenging but possible. What field interests you?"),
        ],
        "Health & Fitness": [
            ("What's a good beginner workout routine?", "Start with 3 days per week: cardio, strength training, and flexibility."),
            ("How many calories should I eat to lose weight?", "It depends on your age, weight, and activity level. Generally, a 500-calorie deficit is safe."),
            ("I'm struggling with motivation to exercise", "Try finding activities you enjoy - dancing, hiking, or sports with friends."),
        ],
        "Language Learning": [
            ("What's the best way to learn Spanish?", "Immersion is key - try Spanish podcasts, movies, and conversation practice."),
            ("How long does it take to become fluent?", "Fluency typically takes 2-3 years of consistent daily practice."),
            ("I keep forgetting vocabulary words", "Try spaced repetition with flashcards and use new words in sentences."),
        ]
    }
    
    conversations = []
    categories = list(templates.keys())
    
    for i in range(count):
        # Pick random category and template
        category = random.choice(categories)
        user_msg, assistant_msg = random.choice(templates[category])
        
        # Add some variation
        if random.random() < 0.3:  # 30% chance of follow-up
            follow_ups = [
                ("Thanks, that's helpful!", "You're welcome! Let me know if you need anything else."),
                ("Can you explain more about that?", "Of course! Here are some additional details..."),
                ("I'm still confused", "No problem, let me try explaining it differently."),
            ]
            user_followup, assistant_followup = random.choice(follow_ups)
            
            messages = [
                Message(role="user", content=user_msg),
                Message(role="assistant", content=assistant_msg),
                Message(role="user", content=user_followup),
                Message(role="assistant", content=assistant_followup),
            ]
        else:
            messages = [
                Message(role="user", content=user_msg),
                Message(role="assistant", content=assistant_msg),
            ]
        
        conv = ConversationData(
            messages=messages,
            metadata={"category": category, "generated": True}
        )
        conversations.append(conv)
    
    return conversations


# Make CLI the default if run directly
if __name__ == '__main__':
    # If no arguments provided, default to UI
    import sys
    if len(sys.argv) == 1:
        sys.argv.append('ui')
    cli()