#!/usr/bin/env python3
"""
Simplified BriefX CLI for testing
"""

import sys
import json
from pathlib import Path
import click
from rich.console import Console

# Add the current directory to Python path
sys.path.insert(0, '.')

from briefx.data.parsers import parse_json_conversations
from briefx.examples import generate_example_conversations

console = Console()

@click.group()
def cli():
    """BriefX - Conversation Analysis Platform"""
    pass

@cli.command()
@click.option('--count', '-c', default=5, help='Number of conversations to generate')
@click.option('--output', '-o', default='examples.json', help='Output file')
def generate(count, output):
    """Generate example conversations"""
    console.print(f"[blue]Generating {count} example conversations...[/blue]")
    
    conversations = generate_example_conversations(count=count, seed=42)
    
    # Save to file - format that parser expects
    output_data = [
        {
            'messages': [
                {'role': msg.role, 'content': msg.content}
                for msg in conv.messages
            ],
            'metadata': conv.metadata
        }
        for conv in conversations
    ]
    
    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"[green]✅ Generated {len(conversations)} conversations saved to {output}[/green]")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def analyze(input_file):
    """Analyze conversations from file"""
    console.print(f"[blue]Analyzing conversations from {input_file}...[/blue]")
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    conversations = parse_json_conversations(content)
    
    console.print(f"[green]✅ Parsed {len(conversations)} conversations[/green]")
    
    for i, conv in enumerate(conversations[:3]):  # Show first 3
        console.print(f"  Conversation {i+1}: {len(conv.messages)} messages")

@cli.command() 
@click.option('--port', '-p', default=8080, help='Server port')
def serve(port):
    """Start the web server"""
    console.print(f"[blue]Starting server on port {port}...[/blue]")
    console.print(f"[green]Open http://localhost:{port} in your browser[/green]")
    console.print("[yellow]Note: Use 'python app.py' for full server functionality[/yellow]")

@cli.command()
def test():
    """Run basic functionality tests"""
    console.print("[blue]Running basic tests...[/blue]")
    
    # Test example generation
    conversations = generate_example_conversations(count=2, seed=123)
    console.print(f"✅ Generated {len(conversations)} test conversations")
    
    # Test conversation structure
    if conversations:
        conv = conversations[0]
        console.print(f"✅ First conversation: {len(conv.messages)} messages")
        console.print(f"   Category: {conv.metadata.get('category')}")
        console.print(f"   Topic: {conv.metadata.get('topic')}")
    
    console.print("[green]🎉 Basic tests passed![/green]")

if __name__ == '__main__':
    cli()