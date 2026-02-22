"""
Advanced Prompting System for conversation analysis

Provides comprehensive prompt management with:
- Template-based prompt generation
- Dynamic prompt construction
- Version control for prompts
- A/B testing framework
- Performance metrics tracking
- Prompt optimization
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import uuid
import re

from ..data.models import ConversationData, Message

logger = logging.getLogger(__name__)

# ============================================================================
# Prompt Types and Templates
# ============================================================================

class PromptType(Enum):
    """Types of prompts in the system"""
    FACET_EXTRACTION = "facet_extraction"
    CLUSTER_NAMING = "cluster_naming"
    HIERARCHY_NAMING = "hierarchy_naming"
    DEDUPLICATION = "deduplication"
    CLUSTER_ASSIGNMENT = "cluster_assignment"
    CATEGORY_REFINEMENT = "category_refinement"
    SUMMARIZATION = "summarization"
    CONVERSATION_ANALYSIS = "conversation_analysis"
    PATTERN_DISCOVERY = "pattern_discovery"
    INSIGHT_GENERATION = "insight_generation"

class PromptVersion(Enum):
    """Prompt versions for A/B testing"""
    V1_BASELINE = "v1_baseline"
    V2_ENHANCED = "v2_enhanced"
    V3_OPTIMIZED = "v3_optimized"
    EXPERIMENTAL = "experimental"

@dataclass
class PromptTemplate:
    """Prompt template with metadata"""
    id: str
    name: str
    type: PromptType
    version: PromptVersion
    template: str
    variables: List[str]
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    usage_count: int = 0
    success_rate: float = 0.0

@dataclass
class PromptExecution:
    """Track prompt execution for metrics"""
    id: str
    template_id: str
    prompt_type: PromptType
    version: PromptVersion
    input_data: Dict[str, Any]
    generated_prompt: str
    response: Optional[str] = None
    execution_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    success: bool = False
    error_message: Optional[str] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))

# ============================================================================
# Core Prompt Templates
# ============================================================================

class PromptTemplates:
    """Core prompt templates matching Rust implementation"""
    
    FACET_EXTRACTION = """<conversation>
{conversation}
</conversation>

{question}

Answer:"""

    CLUSTER_NAMING = """You are an expert at summarizing groups of related items. Below are examples of "{facet_name}" values that form a coherent group, along with some examples that are NOT in this group.

Examples IN this group:
{cluster_examples}

Examples NOT in this group:
{non_cluster_examples}

Based on the examples that ARE in this group, provide:
1. A clear, specific name for this group (2-10 words)
2. A brief description of what unites these examples (1-2 sentences){summary_criteria}

Format your response as:
Name: [your cluster name]
Description: [your description]"""

    HIERARCHY_NAMING = """You are organizing clusters of "{facet_name}" into higher-level categories. Below are {num_clusters} existing clusters:

{clusters_text}

Create {num_categories} higher-level categories that group these clusters thematically. Each category should:
1. Have a clear, descriptive name (3-10 words)
2. Have a brief description explaining the theme (1-2 sentences)
3. Be broad enough to contain multiple clusters but specific enough to be meaningful

Format your response as a numbered list:
1. Name: [category name]
   Description: [category description]
2. Name: [category name]
   Description: [category description]
..."""

    DEDUPLICATION = """Review these category names and descriptions for duplicates or near-duplicates:

{categories_text}

Identify which categories are essentially the same and should be merged. For each group of duplicates, provide:
1. The indices of the duplicate categories (comma-separated)
2. A single merged name and description

Format your response as:
Merge [indices]: Name: [merged name], Description: [merged description]

If there are no duplicates, respond with: "No duplicates found\""""

    CLUSTER_ASSIGNMENT = """Assign each cluster to the most appropriate higher-level category.

Clusters:
{clusters_text}

Categories:
{categories_text}

For each cluster, provide its assignment in the format:
1 -> A
2 -> B
...

Every cluster must be assigned to exactly one category."""

    CATEGORY_REFINEMENT = """Review this category and its contents, then provide an improved name and description if needed.

Current Category:
Name: {category_name}
Description: {category_description}

Contains these clusters:
{children_text}

Based on what's actually in this category, provide:
1. A name that accurately reflects the contents (3-10 words)
2. A description that captures what unites these clusters (1-2 sentences)

Format your response as:
Name: [improved name]
Description: [improved description]"""

    SUMMARIZATION = """Summarize the following data concisely:

{data}

Provide a clear, informative summary in 1-3 sentences."""

    # Enhanced templates for V2
    FACET_EXTRACTION_V2 = """<system>You are analyzing a conversation to extract specific information. Be precise and factual.</system>

<conversation>
{conversation}
</conversation>

<task>
{question}
</task>

<instructions>
- Provide a direct answer based only on the conversation content
- If the information is not available, respond with "Not available"
- Be concise but complete
</instructions>

Answer:"""

    CONVERSATION_ANALYSIS = """Analyze the following conversation for key insights:

<conversation>
{conversation}
</conversation>

Identify:
1. Main topics discussed (up to 5)
2. Sentiment/tone of the conversation
3. Key decisions or outcomes (if any)
4. Unresolved issues (if any)
5. Action items or next steps (if any)

Format your response as structured JSON."""

    PATTERN_DISCOVERY = """Analyze these conversation samples to identify patterns:

<samples>
{samples}
</samples>

Identify:
1. Common themes or topics (list up to 5)
2. Recurring patterns in communication style
3. Frequently mentioned concepts or entities
4. Common problems or pain points
5. Typical resolutions or outcomes

Provide a structured analysis with specific examples."""

    INSIGHT_GENERATION = """Based on the following analysis results, generate actionable insights:

<clusters>
{clusters}
</clusters>

<patterns>
{patterns}
</patterns>

<statistics>
{statistics}
</statistics>

Generate 3-5 actionable insights that:
1. Are specific and measurable
2. Address identified patterns or trends
3. Provide clear recommendations
4. Include potential impact

Format each insight as:
Insight: [title]
Observation: [what was found]
Recommendation: [what to do]
Impact: [expected outcome]"""

# ============================================================================
# Advanced Prompt Manager
# ============================================================================

class AdvancedPromptManager:
    """Advanced prompt management system with versioning and optimization"""
    
    def __init__(self, persistence_manager=None):
        self.persistence_manager = persistence_manager
        self.templates: Dict[str, PromptTemplate] = {}
        self.executions: List[PromptExecution] = []
        self.ab_test_groups: Dict[str, List[PromptVersion]] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default prompt templates"""
        # V1 Baseline templates
        self._register_template(
            "facet_extraction_v1",
            PromptType.FACET_EXTRACTION,
            PromptVersion.V1_BASELINE,
            PromptTemplates.FACET_EXTRACTION,
            ["conversation", "question"],
            "Basic facet extraction from conversations"
        )
        
        self._register_template(
            "cluster_naming_v1",
            PromptType.CLUSTER_NAMING,
            PromptVersion.V1_BASELINE,
            PromptTemplates.CLUSTER_NAMING,
            ["facet_name", "cluster_examples", "non_cluster_examples", "summary_criteria"],
            "Generate names for clusters based on examples"
        )
        
        self._register_template(
            "hierarchy_naming_v1",
            PromptType.HIERARCHY_NAMING,
            PromptVersion.V1_BASELINE,
            PromptTemplates.HIERARCHY_NAMING,
            ["facet_name", "num_clusters", "clusters_text", "num_categories"],
            "Create hierarchical category names"
        )
        
        # V2 Enhanced templates
        self._register_template(
            "facet_extraction_v2",
            PromptType.FACET_EXTRACTION,
            PromptVersion.V2_ENHANCED,
            PromptTemplates.FACET_EXTRACTION_V2,
            ["conversation", "question"],
            "Enhanced facet extraction with better instructions",
            system_prompt="You are a precise conversation analyst."
        )
        
        self._register_template(
            "conversation_analysis_v2",
            PromptType.CONVERSATION_ANALYSIS,
            PromptVersion.V2_ENHANCED,
            PromptTemplates.CONVERSATION_ANALYSIS,
            ["conversation"],
            "Comprehensive conversation analysis",
            temperature=0.5
        )
        
        self._register_template(
            "pattern_discovery_v2",
            PromptType.PATTERN_DISCOVERY,
            PromptVersion.V2_ENHANCED,
            PromptTemplates.PATTERN_DISCOVERY,
            ["samples"],
            "Discover patterns across conversations",
            temperature=0.6
        )
        
        self._register_template(
            "insight_generation_v2",
            PromptType.INSIGHT_GENERATION,
            PromptVersion.V2_ENHANCED,
            PromptTemplates.INSIGHT_GENERATION,
            ["clusters", "patterns", "statistics"],
            "Generate actionable insights from analysis",
            temperature=0.7
        )
    
    def _register_template(
        self,
        template_id: str,
        prompt_type: PromptType,
        version: PromptVersion,
        template: str,
        variables: List[str],
        description: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """Register a prompt template"""
        self.templates[template_id] = PromptTemplate(
            id=template_id,
            name=template_id,
            type=prompt_type,
            version=version,
            template=template,
            variables=variables,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def get_template(
        self,
        prompt_type: PromptType,
        version: Optional[PromptVersion] = None
    ) -> Optional[PromptTemplate]:
        """Get a prompt template by type and version"""
        if version:
            # Get specific version
            for template in self.templates.values():
                if template.type == prompt_type and template.version == version:
                    return template
        else:
            # Get best performing version
            best_template = None
            best_score = -1.0
            
            for template in self.templates.values():
                if template.type == prompt_type:
                    score = template.success_rate * (1 + template.usage_count / 100)
                    if score > best_score:
                        best_score = score
                        best_template = template
            
            return best_template
        
        return None
    
    def generate_prompt(
        self,
        prompt_type: PromptType,
        variables: Dict[str, Any],
        version: Optional[PromptVersion] = None,
        use_ab_testing: bool = False
    ) -> Tuple[str, PromptTemplate]:
        """Generate a prompt from template"""
        # Select template based on A/B testing or version
        if use_ab_testing and prompt_type.value in self.ab_test_groups:
            version = self._select_ab_test_version(prompt_type)
        
        template = self.get_template(prompt_type, version)
        if not template:
            raise ValueError(f"No template found for {prompt_type.value}")
        
        # Validate variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing variables for template: {missing_vars}")
        
        # Generate prompt
        try:
            generated_prompt = template.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")
        
        # Track usage
        template.usage_count += 1
        template.updated_at = int(time.time())
        
        return generated_prompt, template
    
    def _select_ab_test_version(self, prompt_type: PromptType) -> PromptVersion:
        """Select version for A/B testing using epsilon-greedy strategy"""
        import random
        
        epsilon = 0.1  # Exploration rate
        versions = self.ab_test_groups.get(prompt_type.value, [])
        
        if not versions or random.random() < epsilon:
            # Explore: random selection
            all_versions = [t.version for t in self.templates.values() if t.type == prompt_type]
            return random.choice(all_versions) if all_versions else PromptVersion.V1_BASELINE
        else:
            # Exploit: select best performing
            best_version = None
            best_score = -1.0
            
            for version in versions:
                template = self.get_template(prompt_type, version)
                if template and template.success_rate > best_score:
                    best_score = template.success_rate
                    best_version = version
            
            return best_version or PromptVersion.V1_BASELINE
    
    def track_execution(
        self,
        template: PromptTemplate,
        input_data: Dict[str, Any],
        generated_prompt: str,
        response: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
        success: bool = False,
        error_message: Optional[str] = None
    ) -> PromptExecution:
        """Track prompt execution for metrics"""
        execution = PromptExecution(
            id=str(uuid.uuid4()),
            template_id=template.id,
            prompt_type=template.type,
            version=template.version,
            input_data=input_data,
            generated_prompt=generated_prompt,
            response=response,
            execution_time_ms=execution_time_ms,
            tokens_used=tokens_used,
            success=success,
            error_message=error_message
        )
        
        self.executions.append(execution)
        
        # Update template metrics
        total_executions = sum(1 for e in self.executions if e.template_id == template.id)
        successful_executions = sum(1 for e in self.executions if e.template_id == template.id and e.success)
        template.success_rate = successful_executions / max(1, total_executions)
        
        # Update performance metrics
        if execution_time_ms:
            template.performance_metrics['avg_execution_time'] = (
                template.performance_metrics.get('avg_execution_time', 0) * 0.9 + execution_time_ms * 0.1
            )
        
        if tokens_used:
            template.performance_metrics['avg_tokens'] = (
                template.performance_metrics.get('avg_tokens', 0) * 0.9 + tokens_used * 0.1
            )
        
        return execution
    
    def optimize_template(
        self,
        template_id: str,
        optimization_goal: str = "success_rate"
    ) -> Optional[PromptTemplate]:
        """Optimize a template based on execution history"""
        template = self.templates.get(template_id)
        if not template:
            return None
        
        # Collect execution data
        executions = [e for e in self.executions if e.template_id == template_id]
        if len(executions) < 10:
            logger.info(f"Not enough data to optimize template {template_id}")
            return template
        
        # Analyze patterns in successful executions
        successful = [e for e in executions if e.success]
        failed = [e for e in executions if not e.success]
        
        if optimization_goal == "success_rate" and len(successful) > 0:
            # Find common patterns in successful prompts
            # This is a simplified optimization - in production, use ML techniques
            common_patterns = self._find_common_patterns(successful)
            
            # Create optimized version
            optimized_template = PromptTemplate(
                id=f"{template_id}_optimized",
                name=f"{template.name} (Optimized)",
                type=template.type,
                version=PromptVersion.V3_OPTIMIZED,
                template=self._apply_optimizations(template.template, common_patterns),
                variables=template.variables,
                description=f"Optimized version of {template.description}",
                system_prompt=template.system_prompt,
                temperature=template.temperature * 0.9,  # Slightly lower temperature
                max_tokens=template.max_tokens
            )
            
            self.templates[optimized_template.id] = optimized_template
            return optimized_template
        
        return template
    
    def _find_common_patterns(self, executions: List[PromptExecution]) -> Dict[str, Any]:
        """Find common patterns in successful executions"""
        patterns = {
            'avg_input_length': 0,
            'common_keywords': [],
            'optimal_temperature': 0.7,
            'response_patterns': []
        }
        
        # Analyze input patterns
        total_length = 0
        keyword_counts = {}
        
        for execution in executions:
            prompt_length = len(execution.generated_prompt)
            total_length += prompt_length
            
            # Extract keywords (simplified)
            words = re.findall(r'\b\w+\b', execution.generated_prompt.lower())
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    keyword_counts[word] = keyword_counts.get(word, 0) + 1
        
        patterns['avg_input_length'] = total_length / len(executions) if executions else 0
        patterns['common_keywords'] = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return patterns
    
    def _apply_optimizations(self, template: str, patterns: Dict[str, Any]) -> str:
        """Apply optimizations to template based on patterns"""
        # This is a simplified optimization
        # In production, use more sophisticated NLP techniques
        
        optimized = template
        
        # Add emphasis to common successful keywords
        for keyword, _ in patterns.get('common_keywords', [])[:5]:
            if keyword in template.lower():
                optimized = optimized.replace(keyword, f"**{keyword}**")
        
        # Add clarity improvements
        optimized = optimized.replace("Provide", "Please provide")
        optimized = optimized.replace("Format your response", "Format your response exactly")
        
        return optimized
    
    def get_prompt(
        self,
        prompt_type_str: str,
        conversation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PromptTemplate:
        """Get a prompt template by type string.

        This is a convenience method used by the pipeline for facet extraction.
        Returns the template object (which has a .version attribute).
        """
        try:
            prompt_type = PromptType(prompt_type_str)
        except ValueError:
            # Try matching by name
            prompt_type = None
            for pt in PromptType:
                if pt.name.lower() == prompt_type_str.lower():
                    prompt_type = pt
                    break
            if prompt_type is None:
                prompt_type = PromptType.FACET_EXTRACTION

        template = self.get_template(prompt_type)
        if template is None:
            # Return the first available template
            template = next(iter(self.templates.values()), None)
        return template

    def record_result(
        self,
        prompt_type_str: str,
        version: PromptVersion,
        execution_time: float,
        success: bool = True,
        **kwargs
    ) -> None:
        """Record the result of a prompt execution.

        Convenience wrapper around track_execution used by the pipeline.
        """
        try:
            prompt_type = PromptType(prompt_type_str)
        except ValueError:
            prompt_type = PromptType.FACET_EXTRACTION

        template = self.get_template(prompt_type, version)
        if template is None:
            template = self.get_template(prompt_type)
        if template is None:
            return

        self.track_execution(
            template=template,
            input_data={},
            generated_prompt="",
            execution_time_ms=int(execution_time * 1000),
            success=success
        )

    def optimize_templates(self, prompt_type_str: str) -> None:
        """Optimize all templates of a given type.

        Convenience wrapper around optimize_template used by the pipeline.
        """
        try:
            prompt_type = PromptType(prompt_type_str)
        except ValueError:
            return

        for template_id, template in list(self.templates.items()):
            if template.type == prompt_type:
                self.optimize_template(template_id)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of prompt metrics"""
        summary = {
            'total_templates': len(self.templates),
            'total_executions': len(self.executions),
            'templates_by_type': {},
            'success_rates': {},
            'performance_metrics': {}
        }
        
        for template in self.templates.values():
            # Count by type
            type_key = template.type.value
            summary['templates_by_type'][type_key] = summary['templates_by_type'].get(type_key, 0) + 1
            
            # Success rates
            if template.usage_count > 0:
                summary['success_rates'][template.id] = template.success_rate
            
            # Performance metrics
            if template.performance_metrics:
                summary['performance_metrics'][template.id] = template.performance_metrics
        
        return summary
    
    def export_templates(self) -> Dict[str, Any]:
        """Export all templates for backup or sharing"""
        return {
            'version': '1.0',
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'templates': [
                {
                    'id': t.id,
                    'name': t.name,
                    'type': t.type.value,
                    'version': t.version.value,
                    'template': t.template,
                    'variables': t.variables,
                    'description': t.description,
                    'system_prompt': t.system_prompt,
                    'temperature': t.temperature,
                    'max_tokens': t.max_tokens,
                    'performance_metrics': t.performance_metrics,
                    'usage_count': t.usage_count,
                    'success_rate': t.success_rate
                }
                for t in self.templates.values()
            ]
        }
    
    def import_templates(self, data: Dict[str, Any]) -> int:
        """Import templates from export"""
        imported = 0
        
        for template_data in data.get('templates', []):
            try:
                template = PromptTemplate(
                    id=template_data['id'],
                    name=template_data['name'],
                    type=PromptType(template_data['type']),
                    version=PromptVersion(template_data['version']),
                    template=template_data['template'],
                    variables=template_data['variables'],
                    description=template_data.get('description'),
                    system_prompt=template_data.get('system_prompt'),
                    temperature=template_data.get('temperature', 0.7),
                    max_tokens=template_data.get('max_tokens'),
                    performance_metrics=template_data.get('performance_metrics', {}),
                    usage_count=template_data.get('usage_count', 0),
                    success_rate=template_data.get('success_rate', 0.0)
                )
                
                self.templates[template.id] = template
                imported += 1
                
            except Exception as e:
                logger.error(f"Failed to import template {template_data.get('id')}: {e}")
        
        return imported

# ============================================================================
# Utility Functions (matching Rust implementation)
# ============================================================================

def conversation_to_string(conversation: ConversationData) -> str:
    """Convert conversation to string format"""
    return "\n".join([
        f"{msg.role}: {msg.content}"
        for msg in conversation.messages
    ])

def get_facet_prompt(conversation: ConversationData, facet_question: str) -> str:
    """Generate facet extraction prompt"""
    manager = AdvancedPromptManager()
    prompt, _ = manager.generate_prompt(
        PromptType.FACET_EXTRACTION,
        {
            'conversation': conversation_to_string(conversation),
            'question': facet_question
        }
    )
    return prompt

def get_cluster_name_prompt(
    facet_name: str,
    cluster_examples: List[str],
    non_cluster_examples: List[str],
    summary_criteria: Optional[str] = None
) -> str:
    """Generate cluster naming prompt"""
    manager = AdvancedPromptManager()
    
    cluster_text = "\n".join([
        f"{i+1}: {ex}" for i, ex in enumerate(cluster_examples)
    ])
    
    non_cluster_text = "\n".join([
        f"{i+1}: {ex}" for i, ex in enumerate(non_cluster_examples)
    ])
    
    summary_criteria_text = f"\n\n{summary_criteria}" if summary_criteria else ""
    
    prompt, _ = manager.generate_prompt(
        PromptType.CLUSTER_NAMING,
        {
            'facet_name': facet_name,
            'cluster_examples': cluster_text,
            'non_cluster_examples': non_cluster_text,
            'summary_criteria': summary_criteria_text
        }
    )
    return prompt

def get_hierarchy_prompt(
    facet_name: str,
    cluster_summaries: List[Tuple[str, str]],
    n_desired_names: int
) -> str:
    """Generate hierarchy naming prompt"""
    manager = AdvancedPromptManager()
    
    clusters_text = "\n".join([
        f"{i+1}: {name} - {desc}"
        for i, (name, desc) in enumerate(cluster_summaries)
    ])
    
    prompt, _ = manager.generate_prompt(
        PromptType.HIERARCHY_NAMING,
        {
            'facet_name': facet_name,
            'num_clusters': len(cluster_summaries),
            'clusters_text': clusters_text,
            'num_categories': n_desired_names
        }
    )
    return prompt

# ============================================================================
# Global prompt manager instance
# ============================================================================

_prompt_manager: Optional[AdvancedPromptManager] = None

def get_prompt_manager() -> AdvancedPromptManager:
    """Get global prompt manager instance"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = AdvancedPromptManager()
    return _prompt_manager

def initialize_prompt_manager(persistence_manager=None) -> AdvancedPromptManager:
    """Initialize global prompt manager"""
    global _prompt_manager
    _prompt_manager = AdvancedPromptManager(persistence_manager=persistence_manager)
    return _prompt_manager