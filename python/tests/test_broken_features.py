"""
Comprehensive tests to identify all broken features in BriefXAI.

Each test targets a specific feature area and documents what's broken.
"""

import pytest
import sys
import os
import json
import importlib

# Ensure the python directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# 1. IMPORT / STARTUP TESTS
# ============================================================================

class TestImportsAndStartup:
    """Test that all modules can be imported without errors."""

    def test_import_data_models(self):
        """Core data models should import cleanly."""
        from briefx.data.models import (
            ConversationData, Message, ConversationCluster,
            FacetValue, ConversationAnalysis, AnalysisResults
        )
        assert ConversationData is not None

    def test_import_parsers(self):
        """Parsers should import cleanly."""
        from briefx.data.parsers import (
            parse_json_conversations, parse_csv_conversations,
            parse_text_conversations, detect_file_format
        )
        assert parse_json_conversations is not None

    def test_import_utils(self):
        """Utils should import cleanly."""
        from briefx.utils import (
            setup_logging, dedup_conversations,
            calculate_simple_sentiment, determine_category,
            generate_session_id
        )
        assert setup_logging is not None

    def test_import_session_manager(self):
        """Session manager should import cleanly."""
        from briefx.analysis.session_manager import SessionManager, SessionInfo
        assert SessionManager is not None

    def test_import_clustering(self):
        """Clustering should import cleanly."""
        from briefx.analysis.clustering import ConversationClusterer, ClusterEvaluator
        assert ConversationClusterer is not None

    def test_import_dimensionality(self):
        """Dimensionality reduction should import cleanly."""
        from briefx.analysis.dimensionality import (
            reduce_embeddings_for_clustering,
            reduce_embeddings_for_visualization
        )
        assert reduce_embeddings_for_clustering is not None

    def test_import_clio(self):
        """Clio module should import cleanly."""
        from briefx.analysis.clio import (
            ClioAnalysisPipeline, ClioAnalysisRequest,
            PrivacyConfig, ClusteringConfig, PIIDetector
        )
        assert ClioAnalysisPipeline is not None

    def test_import_providers_factory(self):
        """Provider factory should import cleanly."""
        from briefx.providers.factory import (
            ProviderFactory, get_available_providers,
            LlmProviderEnum, EmbeddingProviderEnum
        )
        assert ProviderFactory is not None

    def test_import_providers_base(self):
        """Provider base classes should import cleanly."""
        from briefx.providers.base import LLMProvider, EmbeddingProvider
        assert LLMProvider is not None

    def test_import_preprocessing(self):
        """Preprocessing should import cleanly."""
        from briefx.preprocessing import SmartPreprocessor, PreprocessingOptions
        assert SmartPreprocessor is not None

    def test_import_persistence(self):
        """Persistence should import cleanly."""
        from briefx.persistence import (
            EnhancedPersistenceManager, initialize_persistence,
            TemplateCategory
        )
        assert EnhancedPersistenceManager is not None

    def test_import_monitoring(self):
        """Monitoring should import cleanly."""
        from briefx.monitoring import MonitoringSystem, monitoring_system
        assert monitoring_system is not None

    def test_import_error_recovery(self):
        """Error recovery should import cleanly."""
        from briefx.error_recovery import ErrorRecoverySystem, error_recovery_system
        assert error_recovery_system is not None

    def test_import_examples(self):
        """Examples module should import cleanly."""
        from briefx.examples import generate_example_conversations
        assert generate_example_conversations is not None

    def test_import_prompts(self):
        """Prompts module should import cleanly."""
        from briefx.prompts import AdvancedPromptManager
        assert AdvancedPromptManager is not None

    def test_import_pipeline(self):
        """BUG: Pipeline module fails to import because it references a
        non-existent 'config' module (from config import BriefXConfig)."""
        try:
            from briefx.analysis.pipeline import AnalysisPipeline, initialize_pipeline, get_pipeline
            imported = True
        except (ImportError, ModuleNotFoundError) as e:
            imported = False
            error_msg = str(e)
        assert imported, f"Pipeline import failed: {error_msg}"

    def test_app_module_imports(self):
        """BUG: app.py fails to load because pipeline.py has broken imports,
        and app.py calls initialize_pipeline() without required arguments."""
        original_cwd = os.getcwd()
        try:
            os.chdir(os.path.join(os.path.dirname(__file__), '..'))
            import importlib
            # Try importing app module
            spec = importlib.util.spec_from_file_location(
                "app", os.path.join(os.path.dirname(__file__), '..', 'app.py')
            )
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            loaded = True
        except Exception as e:
            loaded = False
            error_msg = str(e)
        finally:
            os.chdir(original_cwd)
        assert loaded, f"App module failed to load: {error_msg}"


# ============================================================================
# 2. PROVIDER FACTORY TESTS
# ============================================================================

class TestProviderFactory:
    """Test provider factory and enum completeness."""

    def test_gemini_enum_exists(self):
        """BUG: LlmProviderEnum is missing GEMINI value, yet the factory
        checks for LlmProviderEnum.GEMINI in create_llm_provider."""
        from briefx.providers.factory import LlmProviderEnum
        assert hasattr(LlmProviderEnum, 'GEMINI'), \
            "LlmProviderEnum is missing GEMINI enum value"

    def test_create_llm_provider_all_types(self):
        """All provider types defined in the enum should be handled by the factory."""
        from briefx.providers.factory import ProviderFactory, LlmProviderEnum
        # Every enum member should be creatable (even if it returns None for missing keys)
        for provider in LlmProviderEnum:
            try:
                result = ProviderFactory.create_llm_provider(
                    provider, api_key="test-key", model="test"
                )
                # Result can be None (missing packages) but shouldn't raise AttributeError
            except AttributeError as e:
                pytest.fail(f"Factory can't handle provider {provider.value}: {e}")

    def test_get_available_providers(self):
        """Available providers endpoint should work."""
        from briefx.providers.factory import get_available_providers
        providers = get_available_providers()
        assert isinstance(providers, dict)
        assert 'openai' in providers


# ============================================================================
# 3. SESSION MANAGER TESTS
# ============================================================================

class TestSessionManager:
    """Test session manager compatibility with app.py's expectations."""

    def test_get_session_info_method_exists(self):
        """BUG: app.py calls session_manager.get_session_info() but the
        SessionManager class only has get_session()."""
        from briefx.analysis.session_manager import SessionManager
        sm = SessionManager()
        assert hasattr(sm, 'get_session_info'), \
            "SessionManager missing get_session_info() method (app.py expects it)"

    def test_session_info_has_current_message(self):
        """BUG: app.py accesses session_info.current_message but SessionInfo
        has current_step instead."""
        from briefx.analysis.session_manager import SessionInfo, SessionStatus
        from datetime import datetime
        info = SessionInfo(
            session_id="test",
            status=SessionStatus.RUNNING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            current_step="Testing"
        )
        assert hasattr(info, 'current_message'), \
            "SessionInfo missing current_message attribute (app.py expects it)"

    def test_session_create_and_get(self):
        """Basic session lifecycle should work."""
        from briefx.analysis.session_manager import SessionManager
        sm = SessionManager()
        session = sm.create_session("test-123", total_steps=10)
        assert session.session_id == "test-123"
        retrieved = sm.get_session("test-123")
        assert retrieved is not None
        assert retrieved.session_id == "test-123"

    def test_session_progress_update(self):
        """Session progress updates should work."""
        from briefx.analysis.session_manager import SessionManager
        sm = SessionManager()
        sm.create_session("test-progress", total_steps=100)
        sm.start_session("test-progress")
        result = sm.update_progress("test-progress", 50.0, "Halfway there")
        assert result is True
        session = sm.get_session("test-progress")
        assert session.progress == 50.0


# ============================================================================
# 4. PIPELINE TESTS
# ============================================================================

class TestAnalysisPipeline:
    """Test the analysis pipeline initialization and execution."""

    def test_pipeline_initialize_no_args(self):
        """BUG: app.py calls initialize_pipeline() with no args but the
        function requires a BriefXConfig parameter."""
        try:
            from briefx.analysis.pipeline import initialize_pipeline
            # Should be callable with no args (or sensible defaults)
            initialize_pipeline()
            initialized = True
        except (TypeError, ImportError, ModuleNotFoundError) as e:
            initialized = False
            error_msg = str(e)
        assert initialized, f"initialize_pipeline() failed: {error_msg}"

    def test_pipeline_get_after_init(self):
        """get_pipeline() should return a pipeline after initialization."""
        try:
            from briefx.analysis.pipeline import initialize_pipeline, get_pipeline
            initialize_pipeline()
            pipeline = get_pipeline()
            assert pipeline is not None
        except Exception as e:
            pytest.fail(f"get_pipeline() failed after init: {e}")

    def test_prompt_manager_get_prompt_method(self):
        """BUG: pipeline.py calls self.prompt_manager.get_prompt() but
        AdvancedPromptManager only has get_template() and generate_prompt()."""
        from briefx.prompts import AdvancedPromptManager
        pm = AdvancedPromptManager()
        assert hasattr(pm, 'get_prompt'), \
            "AdvancedPromptManager missing get_prompt() method (pipeline.py expects it)"

    def test_prompt_manager_record_result_method(self):
        """BUG: pipeline.py calls self.prompt_manager.record_result() but
        AdvancedPromptManager only has track_execution()."""
        from briefx.prompts import AdvancedPromptManager
        pm = AdvancedPromptManager()
        assert hasattr(pm, 'record_result'), \
            "AdvancedPromptManager missing record_result() method (pipeline.py expects it)"

    def test_prompt_manager_optimize_templates_method(self):
        """BUG: pipeline.py calls self.prompt_manager.optimize_templates() but
        AdvancedPromptManager only has optimize_template() (singular)."""
        from briefx.prompts import AdvancedPromptManager
        pm = AdvancedPromptManager()
        assert hasattr(pm, 'optimize_templates'), \
            "AdvancedPromptManager missing optimize_templates() method (pipeline.py expects it)"


# ============================================================================
# 5. CLIO MODULE TESTS
# ============================================================================

class TestClioModule:
    """Test Clio methodology implementation."""

    def test_conversation_data_content_attribute(self):
        """BUG: clio.py accesses conv_data.content but ConversationData
        doesn't have a .content attribute - it uses get_text()."""
        from briefx.data.models import ConversationData, Message
        conv = ConversationData(
            messages=[Message(role="user", content="Hello world")],
            metadata={}
        )
        assert hasattr(conv, 'content'), \
            "ConversationData missing .content attribute (clio.py expects it)"

    def test_embedding_provider_embed_texts_method(self):
        """BUG: clio.py calls self.embedding_provider.embed_texts() but
        the base class defines generate_embeddings()."""
        from briefx.providers.base import EmbeddingProvider
        # Check that embed_texts is an expected method
        assert hasattr(EmbeddingProvider, 'embed_texts') or \
            'embed_texts' in dir(EmbeddingProvider), \
            "EmbeddingProvider missing embed_texts() method (clio.py expects it)"

    def test_pii_detector(self):
        """PII detector should work correctly."""
        from briefx.analysis.clio import PIIDetector
        detector = PIIDetector()

        # Test email detection
        pii = detector.detect_pii("Contact me at john@example.com")
        assert len(pii['emails']) > 0

        # Test phone detection
        pii = detector.detect_pii("Call 555-123-4567")
        assert len(pii['phones']) > 0

        # Test SSN detection
        pii = detector.detect_pii("My SSN is 123-45-6789")
        assert len(pii['ssns']) > 0

    def test_pii_redaction(self):
        """PII redaction should replace sensitive info."""
        from briefx.analysis.clio import PIIDetector
        detector = PIIDetector()

        text = "Email me at john@example.com"
        redacted, counts = detector.redact_pii(text)
        assert '[EMAIL]' in redacted
        assert 'john@example.com' not in redacted


# ============================================================================
# 6. DATA PARSERS TESTS
# ============================================================================

class TestDataParsers:
    """Test conversation data parsers."""

    def test_json_parser_list(self):
        """JSON parser should handle list of conversations."""
        from briefx.data.parsers import parse_json_conversations
        data = json.dumps([
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hi"}]}
        ])
        result = parse_json_conversations(data)
        assert len(result) == 2

    def test_json_parser_single(self):
        """JSON parser should handle a single conversation object."""
        from briefx.data.parsers import parse_json_conversations
        data = json.dumps(
            {"messages": [{"role": "user", "content": "Hello"}]}
        )
        result = parse_json_conversations(data)
        assert len(result) == 1

    def test_json_parser_bytes(self):
        """JSON parser should handle bytes input."""
        from briefx.data.parsers import parse_json_conversations
        data = json.dumps(
            [{"messages": [{"role": "user", "content": "Hello"}]}]
        ).encode('utf-8')
        result = parse_json_conversations(data)
        assert len(result) == 1

    def test_csv_parser_multi_conversation(self):
        """BUG: CSV parser _process_csv_row passes current_id by value,
        so it can't track conversation boundaries properly."""
        from briefx.data.parsers import parse_csv_conversations
        csv_data = "conv_id,role,content\n1,user,Hello\n1,assistant,Hi there\n2,user,Goodbye\n2,assistant,Bye"
        result = parse_csv_conversations(csv_data)
        # Should have 2 conversations (conv_id 1 and conv_id 2)
        assert len(result) >= 2, \
            f"CSV parser only found {len(result)} conversations, expected at least 2"

    def test_text_parser(self):
        """Text parser should handle plain text input."""
        from briefx.data.parsers import parse_text_conversations
        text = "User: Hello\nAssistant: Hi there\nUser: How are you?"
        result = parse_text_conversations(text)
        assert len(result) == 1
        assert len(result[0].messages) == 3

    def test_detect_format_json(self):
        """Format detection should identify JSON."""
        from briefx.data.parsers import detect_file_format
        assert detect_file_format('{"test": true}') == "json"
        assert detect_file_format(b'{"test": true}', "data.json") == "json"

    def test_detect_format_csv(self):
        """Format detection should identify CSV."""
        from briefx.data.parsers import detect_file_format
        assert detect_file_format("", "data.csv") == "csv"


# ============================================================================
# 7. EXAMPLES MODULE TESTS
# ============================================================================

class TestExamplesModule:
    """Test example conversation generation."""

    def test_generate_example_conversations(self):
        """Should generate the requested number of conversations."""
        from briefx.examples import generate_example_conversations
        convs = generate_example_conversations(count=5, seed=42)
        assert len(convs) == 5
        for conv in convs:
            assert len(conv.messages) >= 4

    def test_save_examples_to_file(self):
        """BUG: save_examples_to_file calls conv.dict() but ConversationData
        is a dataclass, not a Pydantic model. Should use to_dict()."""
        from briefx.examples import generate_example_conversations, save_examples_to_file
        import tempfile
        convs = generate_example_conversations(count=2, seed=42)
        try:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                result = save_examples_to_file(convs, f.name)
            saved = True
        except AttributeError as e:
            saved = False
            error_msg = str(e)
        finally:
            try:
                os.unlink(f.name)
            except:
                pass
        assert saved, f"save_examples_to_file failed: {error_msg}"


# ============================================================================
# 8. FLASK APP ENDPOINT TESTS
# ============================================================================

class TestFlaskApp:
    """Test Flask app endpoints (requires app to load)."""

    @pytest.fixture
    def client(self):
        """Create test client - BUG: this will fail if app can't load."""
        original_cwd = os.getcwd()
        try:
            os.chdir(os.path.join(os.path.dirname(__file__), '..'))
            # Need to manipulate sys.path for the imports to work
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client
        except Exception as e:
            pytest.skip(f"Flask app failed to load: {e}")
        finally:
            os.chdir(original_cwd)

    def test_health_endpoint(self, client):
        """Health check endpoint should return OK."""
        response = client.get('/api/health')
        assert response.status_code == 200

    def test_providers_endpoint(self, client):
        """Providers endpoint should return provider list."""
        response = client.get('/api/providers')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'providers' in data

    def test_example_endpoint(self, client):
        """Example data endpoint should return example data."""
        response = client.get('/api/example')
        assert response.status_code == 200
        data = response.get_json()
        assert 'conversations' in data
        assert 'clusters' in data

    def test_generate_examples_endpoint(self, client):
        """Generate examples endpoint should create conversations."""
        response = client.post('/api/generate-examples',
            json={'count': 3, 'seed': 42}
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['data']['count'] == 3

    def test_process_endpoint(self, client):
        """Process endpoint should categorize conversations."""
        response = client.post('/api/process', json={
            'conversations': [
                {'messages': [
                    {'role': 'user', 'content': 'I found a bug in the system'},
                    {'role': 'assistant', 'content': 'I will look into it'}
                ]},
                {'messages': [
                    {'role': 'user', 'content': 'I love this product, great work!'},
                    {'role': 'assistant', 'content': 'Thank you!'}
                ]}
            ]
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert len(data['data']['conversations']) == 2
        assert len(data['data']['clusters']) > 0

    def test_upload_json_endpoint(self, client):
        """Upload endpoint should parse JSON files."""
        import io
        data = json.dumps([
            {"messages": [{"role": "user", "content": "Test message"}]}
        ])
        response = client.post('/api/upload',
            data={'file': (io.BytesIO(data.encode()), 'test.json')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 200
        result = response.get_json()
        assert result['success'] is True

    def test_preprocess_endpoint(self, client):
        """Preprocess endpoint should process conversations."""
        response = client.post('/api/preprocess', json={
            'conversations': [
                {'messages': [
                    {'role': 'user', 'content': 'Hello world'}
                ]}
            ]
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True

    def test_monitoring_metrics_endpoint(self, client):
        """Monitoring metrics endpoint should return metrics."""
        response = client.get('/api/monitoring/metrics')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True

    def test_monitoring_health_endpoint(self, client):
        """Monitoring health endpoint should return health status."""
        response = client.get('/api/monitoring/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True

    def test_privacy_config_endpoint(self, client):
        """Privacy config endpoint should return default config."""
        response = client.get('/api/clio/privacy/config')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True

    def test_investigation_suggestions_endpoint(self, client):
        """Investigation suggestions endpoint should return suggestions."""
        response = client.get('/api/clio/investigate/suggest')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True

    def test_discovery_recommendations_endpoint(self, client):
        """Discovery recommendations endpoint should return recommendations."""
        response = client.post('/api/clio/discovery/recommendations', json={})
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True

    def test_analyze_endpoint(self, client):
        """Analyze endpoint should process conversations."""
        response = client.post('/api/analyze', json={
            'conversations': [
                {'messages': [
                    {'role': 'user', 'content': 'Help me fix a bug'},
                    {'role': 'assistant', 'content': 'What bug are you seeing?'}
                ]}
            ]
        })
        assert response.status_code == 200
        data = response.get_json()
        # It should return successfully even without LLM providers
        assert 'success' in data

    def test_progress_endpoint(self, client):
        """BUG: Progress endpoint uses get_session_info which doesn't exist."""
        response = client.get('/api/progress/fake-session-id')
        # Should not return 500
        assert response.status_code == 200
        data = response.get_json()
        assert 'success' in data

    def test_session_status_endpoint(self, client):
        """BUG: Session status endpoint uses get_session_info which doesn't exist."""
        response = client.get('/api/session/fake-session-id/status')
        assert response.status_code == 200

    def test_static_file_serving(self, client):
        """BUG: Static files path is wrong - Flask looks for briefxai_ui_data
        relative to python/ but it's at project root."""
        response = client.get('/')
        assert response.status_code == 200, \
            "Root path failed - likely wrong static_folder path"


# ============================================================================
# 9. CLUSTERING TESTS
# ============================================================================

class TestClustering:
    """Test clustering functionality."""

    def test_cluster_by_category(self):
        """Category-based clustering should work without embeddings."""
        from briefx.analysis.clustering import ConversationClusterer
        from briefx.data.models import ConversationData, Message

        clusterer = ConversationClusterer()
        conversations = [
            ConversationData(messages=[Message(role="user", content="I found a bug in the login")]),
            ConversationData(messages=[Message(role="user", content="The app crashed with an error")]),
            ConversationData(messages=[Message(role="user", content="Love this product!")]),
        ]
        clusters = clusterer.cluster_conversations(conversations, [])
        assert len(clusters) >= 1

    def test_kmeans_clustering(self):
        """K-means clustering should work with embeddings."""
        import numpy as np
        from briefx.analysis.clustering import ConversationClusterer
        from briefx.data.models import ConversationData, Message

        clusterer = ConversationClusterer(method="kmeans", max_clusters=3)
        conversations = [
            ConversationData(messages=[Message(role="user", content=f"Message {i}")])
            for i in range(20)
        ]
        # Random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(20, 10).tolist()
        clusters = clusterer.cluster_conversations(conversations, embeddings)
        assert len(clusters) >= 1

    def test_cluster_evaluator(self):
        """Cluster evaluator should return metrics."""
        import numpy as np
        from briefx.analysis.clustering import ClusterEvaluator

        X = np.random.randn(20, 5)
        labels = np.array([0] * 10 + [1] * 10)
        metrics = ClusterEvaluator.evaluate_clustering(X, labels)
        assert 'silhouette_score' in metrics
        assert 'n_clusters' in metrics


# ============================================================================
# 10. UTILITY TESTS
# ============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_sentiment_analysis(self):
        """Sentiment calculation should work correctly."""
        from briefx.utils import calculate_simple_sentiment
        positive = calculate_simple_sentiment("This is great and wonderful")
        negative = calculate_simple_sentiment("This is terrible and broken")
        neutral = calculate_simple_sentiment("The sky is blue")

        assert positive > 0
        assert negative < 0
        assert neutral == 0.0

    def test_category_determination(self):
        """Category determination should classify correctly."""
        from briefx.utils import determine_category
        assert determine_category("I found a bug in the system") == 'Bug Report'
        assert determine_category("Can you help me?") == 'Support'
        assert determine_category("I love this product") == 'Feedback'

    def test_dedup_conversations(self):
        """Deduplication should remove exact duplicates."""
        from briefx.utils import dedup_conversations
        from briefx.data.models import ConversationData, Message

        convs = [
            ConversationData(messages=[Message(role="user", content="Hello")]),
            ConversationData(messages=[Message(role="user", content="Hello")]),
            ConversationData(messages=[Message(role="user", content="Different")]),
        ]
        deduped = dedup_conversations(convs)
        assert len(deduped) == 2

    def test_generate_session_id(self):
        """Session ID generation should produce unique UUIDs."""
        from briefx.utils import generate_session_id
        id1 = generate_session_id()
        id2 = generate_session_id()
        assert id1 != id2
        assert len(id1) == 36  # UUID format


# ============================================================================
# 11. MONITORING TESTS
# ============================================================================

class TestMonitoring:
    """Test monitoring system."""

    def test_record_request(self):
        """Should record request metrics."""
        from briefx.monitoring import MonitoringSystem
        ms = MonitoringSystem()
        ms.record_request(True, 0.5)
        ms.record_request(False, 1.0)
        metrics = ms.get_metrics()
        assert metrics['total_requests'] == 2
        assert metrics['successful_requests'] == 1
        assert metrics['failed_requests'] == 1

    def test_health_check(self):
        """Should perform health check without errors."""
        from briefx.monitoring import MonitoringSystem
        ms = MonitoringSystem()
        result = ms.perform_health_check()
        assert result.status is not None
        assert result.overall_score >= 0


# ============================================================================
# 12. PREPROCESSING TESTS
# ============================================================================

class TestPreprocessing:
    """Test preprocessing module."""

    def test_smart_preprocessor(self):
        """Smart preprocessor should process conversations."""
        from briefx.preprocessing import SmartPreprocessor, PreprocessingOptions
        from briefx.data.models import ConversationData, Message

        preprocessor = SmartPreprocessor()
        options = PreprocessingOptions()
        conversations = [
            ConversationData(messages=[
                Message(role="user", content="  Hello world  "),
                Message(role="assistant", content="Hi there!")
            ])
        ]
        processed, report = preprocessor.preprocess(conversations, options)
        assert len(processed) >= 1
        assert report.total_conversations >= 1
