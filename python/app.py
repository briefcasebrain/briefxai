from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import uuid
from datetime import datetime, timezone
import os
import asyncio
import logging
from typing import List, Dict, Any
import re

# Import our modules
# Configuration will be handled directly in the app
from briefx.data.models import ConversationData, Message
from briefx.data.parsers import parse_json_conversations, parse_csv_conversations, parse_text_conversations, detect_file_format
from briefx.analysis.pipeline import initialize_pipeline, get_pipeline
from briefx.analysis.session_manager import session_manager
from briefx.analysis.clio import ClioAnalysisPipeline, ClioAnalysisRequest, PrivacyConfig, ClusteringConfig, SummarizationLevel
from briefx.providers.factory import get_available_providers, ProviderFactory
from briefx.persistence import get_persistence_manager, initialize_persistence
from briefx.utils import setup_logging
from briefx.preprocessing import SmartPreprocessor, PreprocessingOptions
from briefx.monitoring import monitoring_system
from briefx.error_recovery import error_recovery_system
from briefx.examples import generate_example_conversations

# Initialize configuration
setup_logging(verbose=True, debug=False)
logger = logging.getLogger(__name__)

# Simple configuration for free deployment
class Config:
    def __init__(self):
        # Use demo provider by default (free, no API keys required)
        self.llm_provider = "demo"
        self.llm_model = "demo-analyzer"
        self.embedding_provider = "demo"  # Use demo embedding for free operation
        self.embedding_model = "demo-embeddings"
        
        # API keys - empty by default for free deployment
        self.openai_api_key = os.environ.get('OPENAI_API_KEY', '')
        self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY', '')

config = Config()

# Initialize analysis pipeline
initialize_pipeline()

# Initialize persistence layer
persistence_manager = None

async def init_persistence():
    """Initialize persistence layer"""
    global persistence_manager
    try:
        # Use SQLite for development, PostgreSQL for production
        database_url = os.environ.get('DATABASE_URL')
        use_postgresql = bool(database_url and database_url.startswith('postgresql'))
        
        persistence_manager = await initialize_persistence(
            database_url=database_url,
            use_postgresql=use_postgresql
        )
        logger.info("Enhanced persistence layer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize persistence: {e}")
        # Fall back to basic session manager
        persistence_manager = None

# Use different path for Docker vs local development
# Check if we're running in Docker by looking for /app directory
static_folder = 'ui' if os.path.exists('/app/ui') else '../ui'
app = Flask(__name__, static_folder=static_folder, static_url_path='')
CORS(app)

# Initialize persistence on startup (optional)
with app.app_context():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(init_persistence())
        finally:
            loop.close()
    except Exception as e:
        logger.warning(f"Persistence layer not initialized (optional): {e}")
        persistence_manager = None

# Initialize Clio pipeline
clio_pipeline = None

def get_clio_pipeline():
    """Get or create Clio analysis pipeline"""
    global clio_pipeline
    if clio_pipeline is None:
        # Create providers using factory
        llm_provider = ProviderFactory.create_llm_provider(
            config.llm_provider, 
            api_key=config.openai_api_key or config.anthropic_api_key,
            model=config.llm_model
        )
        
        # Create embedding provider (demo provider works without API keys)
        embedding_provider = ProviderFactory.create_embedding_provider(
            config.embedding_provider,
            api_key=config.openai_api_key if config.embedding_provider != "demo" else "",
            model=config.embedding_model
        )
        
        # Create pipeline if both providers are available
        if llm_provider and embedding_provider:
            clio_pipeline = ClioAnalysisPipeline(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                session_manager=session_manager
            )
    
    return clio_pipeline

# Serve the main UI
@app.route('/')
def index():
    return send_from_directory(static_folder, 'index.html')

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    return "Not found", 404

@app.route('/api/health')
def health():
    return "OK"

# =============================================================================
# Enhanced Persistence API Endpoints
# =============================================================================

@app.route('/api/persistence/sessions')
def get_recent_sessions():
    """Get recent analysis sessions"""
    try:
        if not persistence_manager:
            return jsonify({
                'success': False,
                'error': 'Persistence layer not available'
            }), 503
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            limit = int(request.args.get('limit', 10))
            sessions = loop.run_until_complete(
                persistence_manager.get_recent_sessions(limit=limit)
            )
            
            sessions_data = []
            for session in sessions:
                sessions_data.append({
                    'id': session.id,
                    'created_at': session.created_at,
                    'updated_at': session.updated_at,
                    'status': session.status.value,
                    'total_conversations': session.total_conversations,
                    'processed_conversations': session.processed_conversations,
                    'current_batch': session.current_batch,
                    'total_batches': session.total_batches,
                    'error_message': session.error_message
                })
            
            return jsonify({
                'success': True,
                'data': sessions_data,
                'error': None
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/persistence/sessions/<session_id>')
def get_session_details(session_id):
    """Get detailed session information"""
    try:
        if not persistence_manager:
            return jsonify({
                'success': False,
                'error': 'Persistence layer not available'
            }), 503
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            session = loop.run_until_complete(
                persistence_manager.get_session(session_id)
            )
            
            if not session:
                return jsonify({
                    'success': False,
                    'error': 'Session not found'
                }), 404
            
            # Get batch progress
            batch_progress = loop.run_until_complete(
                persistence_manager.get_batch_progress(session_id)
            )
            
            session_data = {
                'id': session.id,
                'created_at': session.created_at,
                'updated_at': session.updated_at,
                'status': session.status.value,
                'config': session.config,
                'total_conversations': session.total_conversations,
                'processed_conversations': session.processed_conversations,
                'current_batch': session.current_batch,
                'total_batches': session.total_batches,
                'error_message': session.error_message,
                'results': session.results,
                'metadata': session.metadata,
                'batch_progress': [
                    {
                        'batch_number': bp.batch_number,
                        'status': bp.status.value,
                        'started_at': bp.started_at,
                        'completed_at': bp.completed_at,
                        'error_message': bp.error_message,
                        'retry_count': bp.retry_count,
                        'processing_time_ms': bp.processing_time_ms
                    } for bp in batch_progress
                ] if isinstance(batch_progress, list) else []
            }
            
            return jsonify({
                'success': True,
                'data': session_data,
                'error': None
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error fetching session details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/persistence/templates')
def get_analysis_templates():
    """Get analysis templates"""
    try:
        if not persistence_manager:
            return jsonify({
                'success': False,
                'error': 'Persistence layer not available'
            }), 503
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Parse filters
            category = request.args.get('category')
            is_public = request.args.get('is_public')
            
            from briefx.persistence import TemplateCategory
            category_filter = None
            if category:
                try:
                    category_filter = TemplateCategory(category.lower())
                except ValueError:
                    pass
            
            is_public_filter = None
            if is_public is not None:
                is_public_filter = is_public.lower() == 'true'
            
            templates = loop.run_until_complete(
                persistence_manager.get_templates(
                    category=category_filter,
                    is_public=is_public_filter
                )
            )
            
            templates_data = []
            for template in templates:
                templates_data.append({
                    'id': template.id,
                    'name': template.name,
                    'description': template.description,
                    'category': template.category.value,
                    'is_public': template.is_public,
                    'config': template.config,
                    'custom_prompts': template.custom_prompts,
                    'facet_definitions': template.facet_definitions,
                    'created_at': template.created_at,
                    'updated_at': template.updated_at,
                    'created_by': template.created_by
                })
            
            return jsonify({
                'success': True,
                'data': templates_data,
                'error': None
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error fetching templates: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/persistence/templates', methods=['POST'])
def save_analysis_template():
    """Save analysis template"""
    try:
        if not persistence_manager:
            return jsonify({
                'success': False,
                'error': 'Persistence layer not available'
            }), 503
        
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No template data provided'
            }), 400
        
        from briefx.persistence import AnalysisTemplate, TemplateCategory
        
        # Create template object
        template = AnalysisTemplate(
            id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            description=data.get('description'),
            category=TemplateCategory(data.get('category', 'general')),
            is_public=data.get('is_public', False),
            config=data['config'],
            custom_prompts=data.get('custom_prompts'),
            facet_definitions=data.get('facet_definitions'),
            created_by=data.get('created_by')
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(
                persistence_manager.save_template(template)
            )
            
            return jsonify({
                'success': True,
                'data': {'id': template.id},
                'error': None
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error saving template: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/persistence/cache/stats')
def get_cache_stats():
    """Get cache statistics"""
    try:
        if not persistence_manager:
            return jsonify({
                'success': False,
                'error': 'Persistence layer not available'
            }), 503
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            stats = loop.run_until_complete(
                persistence_manager.get_database_stats()
            )
            
            return jsonify({
                'success': True,
                'data': stats,
                'error': None
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error fetching cache stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/persistence/cache/cleanup', methods=['POST'])
def cleanup_cache():
    """Clean up expired cache entries"""
    try:
        if not persistence_manager:
            return jsonify({
                'success': False,
                'error': 'Persistence layer not available'
            }), 503
        
        older_than_hours = int(request.json.get('older_than_hours', 24)) if request.json else 24
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cleaned_count = loop.run_until_complete(
                persistence_manager.cleanup_cache(older_than_hours=older_than_hours)
            )
            
            return jsonify({
                'success': True,
                'data': {'cleaned_entries': cleaned_count},
                'error': None
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# End Enhanced Persistence API Endpoints
# =============================================================================

@app.route('/api/providers')
def providers():
    """Get available LLM providers"""
    try:
        available_providers = get_available_providers()
        return jsonify({
            "success": True,
            "providers": available_providers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_full():
    """Full analysis endpoint with LLM and embeddings"""
    data = request.json
    
    if not data or 'conversations' not in data:
        return jsonify({
            'success': False,
            'data': None,
            'error': 'No conversations provided'
        })
    
    try:
        # Parse conversations
        conversations = []
        for conv_data in data['conversations']:
            messages = []
            for msg in conv_data.get('messages', []):
                messages.append(Message(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', '')
                ))
            
            conversations.append(ConversationData(
                messages=messages,
                metadata=conv_data.get('metadata', {})
            ))
        
        # Run analysis pipeline
        session_id = str(uuid.uuid4())
        
        # Run async analysis in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            pipeline = get_pipeline()
            results = loop.run_until_complete(
                pipeline.analyze_conversations(conversations, session_id)
            )
            
            return jsonify({
                'success': True,
                'data': results.to_dict(),
                'error': None
            })
            
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            'success': False,
            'data': {'session_id': session_id if 'session_id' in locals() else str(uuid.uuid4())},
            'error': str(e)
        })

@app.route('/api/upload', methods=['POST'])
def upload():
    """Handle file uploads and parse conversations"""
    uploaded_files = []
    all_conversations = []
    errors = []
    
    # Process uploaded files
    for file_key in request.files:
        file = request.files[file_key]
        
        try:
            # Read file content
            content = file.read()
            
            # Detect file format
            file_format = detect_file_format(content, file.filename)
            
            # Parse based on format
            try:
                if file_format == "json":
                    conversations = parse_json_conversations(content)
                elif file_format == "csv":
                    conversations = parse_csv_conversations(content)
                elif file_format == "text":
                    conversations = parse_text_conversations(content)
                else:
                    conversations = parse_text_conversations(content)  # Fallback
                
                # Convert to dict format for JSON response
                conv_dicts = []
                for conv in conversations:
                    conv_dict = {
                        'messages': [
                            {'role': msg.role, 'content': msg.content} 
                            for msg in conv.messages
                        ],
                        'metadata': conv.metadata
                    }
                    conv_dicts.append(conv_dict)
                
                all_conversations.extend(conv_dicts)
                
                uploaded_files.append({
                    'name': file.filename,
                    'size': len(content),
                    'mime_type': f'{file_format}',
                    'conversations': len(conversations)
                })
                
            except Exception as parse_error:
                errors.append(f"Error parsing {file.filename}: {str(parse_error)}")
                
        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")
    
    if all_conversations:
        session_id = str(uuid.uuid4())
        return jsonify({
            'success': True,
            'data': {
                'session_id': session_id,
                'files': uploaded_files,
                'conversations': all_conversations,
                'total_conversations': len(all_conversations),
                'warnings': errors
            },
            'error': None
        })
    else:
        return jsonify({
            'success': False,
            'data': None,
            'error': 'No valid conversations found' if errors else 'No files uploaded'
        })

@app.route('/api/process', methods=['POST'])
def process_conversations():
    """Process conversations to extract clusters and patterns"""
    data = request.json
    conversations = data.get('conversations', [])
    
    if not conversations:
        return jsonify({
            'success': False,
            'data': None,
            'error': 'No conversations provided'
        })
    
    # Extract clusters based on content analysis
    clusters = extract_clusters(conversations)
    
    # Process conversations with sentiment and categorization
    processed_conversations = []
    for i, conv in enumerate(conversations):
        # Combine all messages into text
        text = ' '.join([msg['content'] for msg in conv.get('messages', [])])
        
        # Determine category and sentiment
        category = determine_category(text)
        sentiment = calculate_sentiment(text)
        
        processed_conversations.append({
            'id': i + 1,
            'text': text[:200],  # First 200 chars
            'category': category,
            'sentiment': sentiment,
            'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000) - (i * 3600000),
            'messages': len(conv.get('messages', []))
        })
    
    # Calculate statistics
    total_messages = sum(len(conv.get('messages', [])) for conv in conversations)
    avg_messages = total_messages // len(conversations) if conversations else 0
    
    return jsonify({
        'success': True,
        'data': {
            'conversations': processed_conversations,
            'clusters': clusters,
            'statistics': {
                'total_conversations': len(conversations),
                'total_messages': total_messages,
                'avg_messages': avg_messages
            }
        },
        'error': None
    })

def extract_clusters(conversations: List[Dict]) -> List[Dict]:
    """Extract clusters from conversations"""
    category_counts = {}
    
    for conv in conversations:
        text = ' '.join([msg['content'] for msg in conv.get('messages', [])])
        category = determine_category(text)
        category_counts[category] = category_counts.get(category, 0) + 1
    
    clusters = []
    descriptions = {
        'Support': 'Customer support and technical assistance requests',
        'Feature Request': 'User suggestions for new features and improvements',
        'Bug Report': 'Technical issues and system problems',
        'Feedback': 'User feedback and testimonials',
        'Sales': 'Pricing inquiries and subscription questions',
        'Documentation': 'Documentation and API reference requests',
        'General': 'General conversation topics'
    }
    
    for i, (name, count) in enumerate(category_counts.items()):
        clusters.append({
            'id': i + 1,
            'name': name,
            'count': count,
            'description': descriptions.get(name, 'General conversation topics')
        })
    
    # Ensure at least one cluster exists
    if not clusters:
        clusters.append({
            'id': 1,
            'name': 'General',
            'count': len(conversations),
            'description': 'General conversation topics'
        })
    
    return clusters

def determine_category(text: str) -> str:
    """Determine conversation category based on content"""
    text_lower = text.lower()
    
    # Check for keywords to categorize
    if any(word in text_lower for word in ['bug', 'error', 'crash', 'broken', 'fix', 'issue with']):
        return 'Bug Report'
    elif any(word in text_lower for word in ['feature', 'request', 'add', 'implement', 'would be nice']):
        return 'Feature Request'
    elif any(word in text_lower for word in ['help', 'support', 'how do', 'how to', 'problem']):
        return 'Support'
    elif any(word in text_lower for word in ['thank', 'great', 'love', 'awesome', 'excellent']):
        return 'Feedback'
    elif any(word in text_lower for word in ['price', 'cost', 'subscription', 'pay', 'billing']):
        return 'Sales'
    elif any(word in text_lower for word in ['document', 'api', 'guide', 'tutorial', 'reference']):
        return 'Documentation'
    else:
        return 'General'

def calculate_sentiment(text: str) -> float:
    """Calculate simple sentiment score"""
    text_lower = text.lower()
    
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                     'fantastic', 'love', 'perfect', 'awesome', 'helpful', 
                     'thank', 'appreciate', 'happy', 'pleased']
    
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'angry',
                     'frustrated', 'annoyed', 'disappointed', 'broken', 
                     'error', 'crash', 'bug', 'fail', 'wrong']
    
    score = 0
    word_count = 0
    
    for word in positive_words:
        if word in text_lower:
            score += 1
            word_count += 1
    
    for word in negative_words:
        if word in text_lower:
            score -= 1
            word_count += 1
    
    # Normalize to -1.0 to 1.0 range
    if word_count > 0:
        return max(-1.0, min(1.0, score / word_count))
    return 0.0

# =============================================================================
# Clio Methodology API Endpoints
# =============================================================================

@app.route('/api/clio/analyze', methods=['POST'])
def start_clio_analysis():
    """Start Clio privacy-preserving analysis"""
    data = request.json
    
    if not data or 'conversations' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No conversations provided',
            'analysis_id': None
        }), 400
    
    try:
        # Parse conversations
        conversations = []
        for conv_data in data['conversations']:
            messages = []
            for msg in conv_data.get('messages', []):
                messages.append(Message(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', '')
                ))
            
            conversations.append(ConversationData(
                messages=messages,
                metadata=conv_data.get('metadata', {})
            ))
        
        # Create Clio analysis request with parameters
        privacy_config = PrivacyConfig(
            k_anonymity=data.get('k_anonymity', 5),
            remove_pii=data.get('remove_pii', True),
            redact_sensitive=data.get('redact_sensitive', True),
            min_cluster_size=data.get('min_cluster_size', 10),
            privacy_level=data.get('privacy_level', 'standard')
        )
        
        clustering_config = ClusteringConfig(
            algorithm=data.get('clustering_algorithm', 'auto'),
            min_samples=data.get('min_samples', 5),
            eps=data.get('eps', 0.3),
            max_clusters=data.get('max_clusters', 50)
        )
        
        request_obj = ClioAnalysisRequest(
            conversations=conversations,
            privacy_config=privacy_config,
            clustering_config=clustering_config,
            min_cluster_size=data.get('min_cluster_size', 10),
            max_hierarchy_depth=data.get('max_hierarchy_depth', 5),
            pattern_threshold=data.get('pattern_threshold', 0.3)
        )
        
        # Get Clio pipeline
        pipeline = get_clio_pipeline()
        if not pipeline:
            return jsonify({
                'status': 'error',
                'message': 'Clio pipeline not available - check provider configuration',
                'analysis_id': None
            }), 500
        
        # Start analysis in background task
        analysis_id = str(uuid.uuid4())
        
        async def run_analysis():
            try:
                results = await pipeline.analyze(request_obj)
                # Store results in session for retrieval
                session_manager.sessions[analysis_id] = {
                    'status': 'completed',
                    'results': results,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Clio analysis failed: {e}")
                session_manager.sessions[analysis_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        
        # Run analysis in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        def run_in_background():
            loop.run_until_complete(run_analysis())
            loop.close()
        
        import threading
        thread = threading.Thread(target=run_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Clio analysis started successfully',
            'analysis_id': analysis_id
        })
        
    except Exception as e:
        logger.error(f"Failed to start Clio analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'analysis_id': None
        }), 500

@app.route('/api/clio/status/<analysis_id>')
def get_clio_status(analysis_id):
    """Get Clio analysis status"""
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data:
        return jsonify({
            'status': 'not_found',
            'message': 'Analysis not found'
        }), 404
    
    if session_data.get('status') == 'completed':
        results = session_data.get('results')
        if results:
            return jsonify({
                'status': 'completed',
                'total_clusters': len(results.clusters),
                'total_patterns': len(results.patterns),
                'hierarchy_depth': results.hierarchy.depth,
                'total_conversations': results.total_conversations,
                'analysis_cost': results.analysis_cost,
                'privacy_audit': results.privacy_audit
            })
    elif session_data.get('status') == 'failed':
        return jsonify({
            'status': 'failed',
            'message': session_data.get('error', 'Unknown error')
        }), 500
    
    # Check for in-progress status
    progress_info = session_manager.get_session_info(analysis_id)
    if progress_info:
        return jsonify({
            'status': 'in_progress',
            'progress': progress_info.progress,
            'message': progress_info.current_message,
            'stage': progress_info.metadata.get('stage', 'processing')
        })
    
    return jsonify({
        'status': 'not_started',
        'message': 'No analysis has been performed yet'
    })

@app.route('/api/clio/hierarchy/<analysis_id>')
def get_clio_hierarchy(analysis_id):
    """Get Clio hierarchy results"""
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data or session_data.get('status') != 'completed':
        return jsonify({
            'error': 'Analysis not found or not completed'
        }), 404
    
    results = session_data.get('results')
    if not results:
        return jsonify({
            'error': 'Results not available'
        }), 404
    
    return jsonify({
        'hierarchy': {
            'root_id': results.hierarchy.root_id,
            'nodes': results.hierarchy.nodes,
            'edges': results.hierarchy.edges,
            'depth': results.hierarchy.depth,
            'total_nodes': results.hierarchy.total_nodes
        },
        'clusters': [
            {
                'id': cluster.id,
                'name': cluster.name,
                'summary': cluster.summary,
                'size': cluster.size,
                'keywords': cluster.keywords,
                'privacy_level': cluster.privacy_level,
                'quality_metrics': cluster.quality_metrics
            }
            for cluster in results.clusters
        ],
        'conversations': results.total_conversations
    })

@app.route('/api/clio/cluster/<analysis_id>/<cluster_id>')
def get_clio_cluster(analysis_id, cluster_id):
    """Get detailed cluster information"""
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data or session_data.get('status') != 'completed':
        return jsonify({
            'error': 'Analysis not found or not completed'
        }), 404
    
    results = session_data.get('results')
    if not results:
        return jsonify({
            'error': 'Results not available'
        }), 404
    
    # Find cluster by ID
    cluster = None
    for c in results.clusters:
        if c.id == cluster_id:
            cluster = c
            break
    
    if not cluster:
        return jsonify({
            'error': 'Cluster not found'
        }), 404
    
    return jsonify({
        'id': cluster.id,
        'name': cluster.name,
        'summary': cluster.summary,
        'size': cluster.size,
        'centroid': cluster.centroid,
        'conversation_ids': cluster.conversation_ids,
        'keywords': cluster.keywords,
        'sentiment_score': cluster.sentiment_score,
        'privacy_level': cluster.privacy_level,
        'anonymity_group': cluster.anonymity_group,
        'patterns': cluster.patterns,
        'quality_metrics': cluster.quality_metrics,
        'metadata': cluster.metadata
    })

@app.route('/api/clio/patterns/<analysis_id>')
def get_clio_patterns(analysis_id):
    """Get discovered patterns"""
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data or session_data.get('status') != 'completed':
        return jsonify({
            'error': 'Analysis not found or not completed'
        }), 404
    
    results = session_data.get('results')
    if not results:
        return jsonify({
            'error': 'Results not available'
        }), 404
    
    patterns = []
    for pattern in results.patterns:
        patterns.append({
            'id': pattern.id,
            'name': pattern.name,
            'description': pattern.description,
            'support': pattern.support,
            'confidence': pattern.confidence,
            'cluster_ids': pattern.cluster_ids,
            'keywords': pattern.keywords,
            'examples': pattern.examples,
            'privacy_score': pattern.privacy_score,
            'metadata': pattern.metadata
        })
    
    return jsonify({
        'patterns': patterns,
        'total_patterns': len(patterns)
    })

@app.route('/api/clio/search/<analysis_id>')
def search_clio_clusters(analysis_id):
    """Search clusters by query"""
    query = request.args.get('query', '').strip()
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({
            'error': 'Query parameter required'
        }), 400
    
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data or session_data.get('status') != 'completed':
        return jsonify({
            'error': 'Analysis not found or not completed'
        }), 404
    
    results = session_data.get('results')
    if not results:
        return jsonify({
            'error': 'Results not available'
        }), 404
    
    # Simple text search through cluster summaries and names
    search_results = []
    query_lower = query.lower()
    
    for cluster in results.clusters:
        relevance = 0.0
        
        # Check name match
        if query_lower in cluster.name.lower():
            relevance += 1.0
        
        # Check summary match
        if query_lower in cluster.summary.lower():
            relevance += 0.5
        
        # Check keywords match
        for keyword in cluster.keywords:
            if query_lower in keyword.lower():
                relevance += 0.3
        
        if relevance > 0:
            search_results.append({
                'cluster_id': cluster.id,
                'cluster_name': cluster.name,
                'relevance': relevance,
                'size': cluster.size,
                'summary': cluster.summary
            })
    
    # Sort by relevance
    search_results.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Apply limit
    search_results = search_results[:limit]
    
    return jsonify(search_results)

@app.route('/api/clio/export/<analysis_id>/<cluster_id>', methods=['POST'])
def export_clio_cluster(analysis_id, cluster_id):
    """Export cluster data"""
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data or session_data.get('status') != 'completed':
        return jsonify({
            'error': 'Analysis not found or not completed'
        }), 404
    
    results = session_data.get('results')
    if not results:
        return jsonify({
            'error': 'Results not available'
        }), 404
    
    # Find cluster
    cluster = None
    for c in results.clusters:
        if c.id == cluster_id:
            cluster = c
            break
    
    if not cluster:
        return jsonify({
            'error': 'Cluster not found'
        }), 404
    
    # Get conversations in this cluster
    cluster_conversations = []
    for conv_id in cluster.conversation_ids:
        if conv_id < len(results.conversations):
            conv = results.conversations[conv_id]
            cluster_conversations.append({
                'id': conv.id,
                'content': conv.content,
                'metadata': conv.metadata,
                'privacy_level': conv.privacy_level
            })
    
    export_data = {
        'cluster': {
            'id': cluster.id,
            'name': cluster.name,
            'summary': cluster.summary,
            'size': cluster.size,
            'keywords': cluster.keywords,
            'privacy_level': cluster.privacy_level
        },
        'conversations': cluster_conversations,
        'metadata': {
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'total_conversations': len(cluster_conversations),
            'cluster_name': cluster.name,
            'analysis_id': analysis_id
        }
    }
    
    return jsonify(export_data)

@app.route('/api/clio/investigate/<analysis_id>/<cluster_id>', methods=['POST'])
def investigate_clio_cluster(analysis_id, cluster_id):
    """Generate investigation report for cluster"""
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data or session_data.get('status') != 'completed':
        return jsonify({
            'error': 'Analysis not found or not completed'
        }), 404
    
    results = session_data.get('results')
    if not results:
        return jsonify({
            'error': 'Results not available'
        }), 404
    
    # Find cluster
    cluster = None
    for c in results.clusters:
        if c.id == cluster_id:
            cluster = c
            break
    
    if not cluster:
        return jsonify({
            'error': 'Cluster not found'
        }), 404
    
    # Get sample conversations (limit to 5 for investigation)
    sample_conversations = []
    sample_size = min(5, len(cluster.conversation_ids))
    
    for i in range(sample_size):
        conv_id = cluster.conversation_ids[i]
        if conv_id < len(results.conversations):
            conv = results.conversations[conv_id]
            sample_conversations.append({
                'id': conv.id,
                'content': conv.content[:500],  # First 500 chars
                'privacy_level': conv.privacy_level
            })
    
    # Find related patterns
    related_patterns = []
    for pattern in results.patterns:
        if cluster_id in pattern.cluster_ids:
            related_patterns.append({
                'id': pattern.id,
                'name': pattern.name,
                'description': pattern.description,
                'confidence': pattern.confidence,
                'keywords': pattern.keywords
            })
    
    investigation = {
        'cluster': {
            'id': cluster.id,
            'name': cluster.name,
            'summary': cluster.summary,
            'size': cluster.size,
            'privacy_level': cluster.privacy_level
        },
        'sample_conversations': sample_conversations,
        'related_patterns': related_patterns,
        'statistics': {
            'total_conversations': cluster.size,
            'quality_metrics': cluster.quality_metrics,
            'keywords': cluster.keywords[:10]  # Top 10 keywords
        },
        'recommendations': [
            "Review sample conversations for common themes",
            "Check related patterns for actionable insights",
            "Consider creating automated responses for common issues",
            "Monitor cluster growth over time for trend analysis"
        ]
    }
    
    return jsonify(investigation)

# =============================================================================
# Additional Clio Advanced Features (Basic Implementation)
# =============================================================================

@app.route('/api/clio/privacy/config')
def get_privacy_config():
    """Get default privacy configuration"""
    default_config = {
        'k_anonymity': 5,
        'remove_pii': True,
        'redact_sensitive': True,
        'min_cluster_size': 10,
        'privacy_level': 'standard',
        'audit_enabled': True
    }
    return jsonify({
        'success': True,
        'data': default_config,
        'error': None
    })

@app.route('/api/clio/map/create', methods=['POST'])
def create_interactive_map():
    """Create interactive map visualization data"""
    data = request.json
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'No data provided'
        }), 400
    
    # Basic map data structure for frontend
    map_data = {
        'map_id': str(uuid.uuid4()),
        'clusters': data.get('clusters', []),
        'coordinates': data.get('umap_coords', []),
        'facet_data': data.get('facet_data', []),
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    return jsonify({
        'success': True,
        'data': map_data,
        'error': None
    })

@app.route('/api/clio/investigate/suggest')
def get_investigation_suggestions():
    """Get investigation suggestions"""
    suggestions = [
        "Show clusters with high privacy scores",
        "Find clusters with positive sentiment", 
        "Search for support-related conversations",
        "Identify large conversation clusters",
        "Find clusters with diverse topics",
        "Analyze clusters by conversation length",
        "Review clusters with common patterns"
    ]
    
    return jsonify({
        'success': True,
        'data': suggestions,
        'error': None
    })

@app.route('/api/clio/discovery/recommendations', methods=['POST'])
def get_discovery_recommendations():
    """Get discovery recommendations for serendipitous exploration"""
    data = request.json
    
    # Basic recommendation logic
    recommendations = [
        {
            'type': 'similar_pattern',
            'title': 'Explore Similar Conversations',
            'description': 'Clusters with similar patterns to your current selection',
            'confidence': 0.8,
            'cluster_ids': []
        },
        {
            'type': 'unexpected_connection',
            'title': 'Surprising Connections',
            'description': 'Unexpectedly related topics that might provide insights',
            'confidence': 0.6,
            'cluster_ids': []
        },
        {
            'type': 'outlier_analysis',
            'title': 'Unique Conversations',
            'description': 'Outlier conversations that stand out from typical patterns',
            'confidence': 0.7,
            'cluster_ids': []
        }
    ]
    
    return jsonify({
        'success': True,
        'data': recommendations,
        'error': None
    })

@app.route('/api/clio/privacy/audit/<analysis_id>')
def get_privacy_audit_details(analysis_id):
    """Get detailed privacy audit results"""
    session_data = session_manager.sessions.get(analysis_id)
    
    if not session_data or session_data.get('status') != 'completed':
        return jsonify({
            'error': 'Analysis not found or not completed'
        }), 404
    
    results = session_data.get('results')
    if not results:
        return jsonify({
            'error': 'Results not available'  
        }), 404
    
    # Return detailed privacy audit
    privacy_audit = results.privacy_audit
    
    detailed_audit = {
        'audit_summary': privacy_audit,
        'privacy_recommendations': privacy_audit.get('recommendations', []),
        'compliance_score': privacy_audit.get('overall_score', 0.0),
        'detailed_findings': {
            'pii_detection': privacy_audit.get('pii_analysis', {}),
            'anonymity_compliance': privacy_audit.get('anonymity_analysis', {}),
            'cluster_size_compliance': privacy_audit.get('cluster_analysis', {})
        },
        'improvement_suggestions': [
            "Enable PII removal for better privacy protection",
            "Increase k-anonymity parameter to improve group sizes", 
            "Consider raising minimum cluster size for enhanced privacy",
            "Regular privacy audits to ensure ongoing compliance"
        ]
    }
    
    return jsonify(detailed_audit)

# =============================================================================
# End Clio API Endpoints
# =============================================================================

@app.route('/api/progress/<session_id>')
def get_progress(session_id):
    """Get progress for a session"""
    session_info = session_manager.get_session_info(session_id)
    
    if session_info:
        return jsonify({
            'success': True,
            'data': {
                'progress': session_info.progress,
                'message': session_info.current_message,
                'status': 'active' if session_info.progress < 100 else 'completed',
                'metadata': session_info.metadata
            },
            'error': None
        })
    else:
        return jsonify({
            'success': False,
            'data': None,
            'error': 'Session not found'
        })

@app.route('/api/session/<session_id>/status')
def get_session_status(session_id):
    """Get status for a session"""
    session_info = session_manager.get_session_info(session_id)
    
    if session_info:
        return jsonify({
            'session_id': session_id,
            'status': 'active' if session_info.progress < 100 else 'completed',
            'progress': session_info.progress,
            'message': session_info.current_message,
        })
    else:
        return jsonify({
            'session_id': session_id,
            'status': 'not_found',
            'progress': 0,
            'message': 'Session not found'
        })

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess conversation data"""
    try:
        data = request.json
        conversations_data = data.get('conversations', [])
        options = data.get('options', {})
        
        # Convert to ConversationData objects
        conversations = []
        for conv in conversations_data:
            messages = [Message(**msg) for msg in conv.get('messages', [])]
            conversations.append(ConversationData(messages=messages, metadata=conv.get('metadata', {})))
        
        # Create preprocessor with options
        preprocessor = SmartPreprocessor()
        preprocess_options = PreprocessingOptions(**options) if options else PreprocessingOptions()
        
        # Run preprocessing
        processed, quality_report = preprocessor.preprocess(conversations, preprocess_options)
        
        # Record metrics
        monitoring_system.record_component_execution("preprocessing", quality_report.estimated_processing_time, True)
        
        return jsonify({
            'success': True,
            'data': {
                'processed_conversations': [{'messages': [{'role': m.role, 'content': m.content} for m in conv.messages], 'metadata': conv.metadata} for conv in processed],
                'quality_report': {
                    'total_conversations': quality_report.total_conversations,
                    'valid_conversations': quality_report.valid_conversations,
                    'overall_quality_score': quality_report.overall_quality_score,
                    'recommendations': quality_report.recommendations,
                    'auto_fixable_issues': quality_report.auto_fixable_issues
                }
            }
        })
    except Exception as e:
        monitoring_system.record_component_execution("preprocessing", 0, False)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/monitoring/metrics')
def get_monitoring_metrics():
    """Get system monitoring metrics"""
    try:
        metrics = monitoring_system.get_metrics()
        return jsonify({
            'success': True,
            'data': metrics
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/monitoring/health')
def get_health_check():
    """Perform health check"""
    try:
        health = monitoring_system.perform_health_check()
        return jsonify({
            'success': True,
            'data': {
                'status': health.status.value,
                'overall_score': health.overall_score,
                'checks': {k: {'status': v.status.value, 'error_rate': v.error_rate} for k, v in health.checks.items()}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-examples', methods=['POST'])
def generate_examples():
    """Generate example conversations"""
    try:
        data = request.json
        count = data.get('count', 10)
        category = data.get('category', None)
        seed = data.get('seed', None)
        
        conversations = generate_example_conversations(count=count, category=category, seed=seed)
        
        return jsonify({
            'success': True,
            'data': {
                'conversations': [{'messages': [{'role': m.role, 'content': m.content} for m in conv.messages], 'metadata': conv.metadata} for conv in conversations],
                'count': len(conversations)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/providers')
def get_providers():
    """Get available LLM providers and their requirements"""
    try:
        providers = ProviderFactory.get_available_llm_providers()
        return jsonify({
            'success': True,
            'data': providers
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update provider settings with user-provided API keys"""
    try:
        data = request.json
        provider = data.get('provider', 'demo')
        api_key = data.get('api_key', '')
        model = data.get('model', 'default')
        
        # Store settings in session or temporary storage
        # For demo purposes, we'll recreate the clio pipeline with new settings
        global clio_pipeline
        
        # Create new provider configuration
        if provider == 'demo':
            # No API key needed for demo
            api_key = ''
        
        # Recreate clio pipeline with new settings
        llm_provider = ProviderFactory.create_llm_provider(
            provider=provider,
            api_key=api_key,
            model=model
        )
        
        if llm_provider:
            # For now, we'll use a simple embedding fallback
            embedding_provider = None
            try:
                if provider == 'openai' and api_key:
                    from briefx.providers.factory import EmbeddingProviderEnum
                    embedding_provider = ProviderFactory.create_embedding_provider(
                        provider=EmbeddingProviderEnum.OPENAI,
                        api_key=api_key
                    )
            except:
                pass
                
            # Update pipeline
            clio_pipeline = ClioAnalysisPipeline(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                session_manager=session_manager
            )
            
            return jsonify({
                'success': True,
                'message': f'Updated to {provider} provider',
                'data': {
                    'provider': provider,
                    'model': model,
                    'has_embeddings': embedding_provider is not None
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to initialize {provider} provider. Check API key.'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_provider', methods=['POST'])
def update_provider_settings():
    """Update provider settings from the UI"""
    try:
        data = request.get_json()
        
        # Update global configuration
        config.llm_provider = data.get('llm_provider', 'demo')
        config.llm_model = data.get('llm_model', 'demo-analyzer')
        config.embedding_provider = data.get('embedding_provider', 'demo')
        config.embedding_model = data.get('embedding_model', 'demo-embeddings')
        
        # Set API keys if provided
        if config.llm_provider != 'demo':
            api_key = data.get('llm_api_key', '')
            if config.llm_provider == 'openai':
                config.openai_api_key = api_key
            elif config.llm_provider == 'anthropic':
                config.anthropic_api_key = api_key
            elif config.llm_provider == 'gemini':
                config.google_api_key = api_key
        
        if config.embedding_provider == 'openai':
            config.openai_api_key = data.get('embedding_api_key', config.openai_api_key)
        
        # Recreate the pipeline with new settings
        global clio_pipeline
        clio_pipeline = None  # Reset pipeline to force recreation
        
        # Test that the pipeline can be created
        pipeline = get_clio_pipeline()
        if pipeline:
            return jsonify({
                'status': 'success',
                'message': f'Successfully configured {config.llm_provider} provider with {config.embedding_provider} embeddings'
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to create analysis pipeline. Please check your configuration.'
            }), 400
            
    except Exception as e:
        logger.error(f"Failed to update provider settings: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Configuration error: {str(e)}'
        }), 500

@app.route('/api/test_provider', methods=['POST'])
def test_provider_connection():
    """Test connection to the configured provider"""
    try:
        data = request.get_json()
        provider = data.get('llm_provider', 'demo')
        api_key = data.get('llm_api_key', '')
        model = data.get('llm_model', 'demo-analyzer')
        
        if provider == 'demo':
            return jsonify({
                'status': 'success',
                'message': 'Demo provider is ready! No API key required.'
            })
        
        # Try to create and test the provider
        llm_provider = ProviderFactory.create_llm_provider(
            provider=provider,
            api_key=api_key,
            model=model
        )
        
        if llm_provider:
            # Try a simple test completion
            try:
                test_response = llm_provider.complete("Test connection - respond with 'OK'")
                return jsonify({
                    'status': 'success',
                    'message': f'Successfully connected to {provider}! Test response: {test_response[:50]}...'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Provider created but test failed: {str(e)}'
                }), 400
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to create {provider} provider. Check your API key.'
            }), 400
            
    except Exception as e:
        logger.error(f"Provider test failed: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Connection test error: {str(e)}'
        }), 500

@app.route('/api/example')
def get_example_data():
    """Return example data for demo purposes"""
    return jsonify({
        'conversations': [
            {'id': 1, 'text': 'Customer support inquiry about billing', 'category': 'Support', 'sentiment': 0.2, 'timestamp': datetime.now(timezone.utc).timestamp() * 1000 - 18000000},
            {'id': 2, 'text': 'Product feature request for dashboard', 'category': 'Feature Request', 'sentiment': 0.8, 'timestamp': datetime.now(timezone.utc).timestamp() * 1000 - 14400000},
            {'id': 3, 'text': 'Technical issue with login system', 'category': 'Bug Report', 'sentiment': -0.5, 'timestamp': datetime.now(timezone.utc).timestamp() * 1000 - 10800000},
            {'id': 4, 'text': 'Positive feedback about new update', 'category': 'Feedback', 'sentiment': 0.9, 'timestamp': datetime.now(timezone.utc).timestamp() * 1000 - 7200000},
            {'id': 5, 'text': 'Question about pricing plans', 'category': 'Sales', 'sentiment': 0.3, 'timestamp': datetime.now(timezone.utc).timestamp() * 1000 - 3600000}
        ],
        'clusters': [
            {'id': 1, 'name': 'Support Issues', 'count': 15, 'description': 'Customer support and technical problems'},
            {'id': 2, 'name': 'Feature Requests', 'count': 8, 'description': 'User suggestions and feature ideas'},
            {'id': 3, 'name': 'Positive Feedback', 'count': 12, 'description': 'Satisfied customer testimonials'},
            {'id': 4, 'name': 'Sales Inquiries', 'count': 6, 'description': 'Pricing and subscription questions'},
            {'id': 5, 'name': 'Bug Reports', 'count': 9, 'description': 'Technical issues and bugs'}
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)