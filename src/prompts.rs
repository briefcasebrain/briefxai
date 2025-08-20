use crate::types::{ConversationData, Facet};

pub fn conversation_to_string(conversation: &ConversationData) -> String {
    conversation
        .messages.iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn get_facet_prompt(conversation: &ConversationData, facet: &Facet) -> String {
    let conv_str = conversation_to_string(conversation);
    
    format!(
        r#"<conversation>
{}
</conversation>

{}

Answer:"#,
        conv_str,
        facet.question
    )
}

pub fn get_facet_cluster_name_prompt(
    facet: &Facet,
    cluster_examples: &[String],
    non_cluster_examples: &[String],
) -> String {
    let cluster_text = cluster_examples
        .iter()
        .enumerate()
        .map(|(i, ex)| format!("{}: {}", i + 1, ex))
        .collect::<Vec<_>>()
        .join("\n");
    
    let non_cluster_text = non_cluster_examples
        .iter()
        .enumerate()
        .map(|(i, ex)| format!("{}: {}", i + 1, ex))
        .collect::<Vec<_>>()
        .join("\n");
    
    let summary_criteria = facet.summary_criteria.as_ref()
        .map(|s| format!("\n\n{}", s))
        .unwrap_or_default();
    
    format!(
        r#"You are an expert at summarizing groups of related items. Below are examples of "{}" values that form a coherent group, along with some examples that are NOT in this group.

Examples IN this group:
{}

Examples NOT in this group:
{}

Based on the examples that ARE in this group, provide:
1. A clear, specific name for this group (2-10 words)
2. A brief description of what unites these examples (1-2 sentences){}

Format your response as:
Name: [your cluster name]
Description: [your description]"#,
        facet.name,
        cluster_text,
        non_cluster_text,
        summary_criteria
    )
}

pub fn get_neighborhood_cluster_names_prompt(
    facet: &Facet,
    cluster_summaries: &[(String, String)],
    n_desired_names: usize,
) -> String {
    let clusters_text = cluster_summaries
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| format!("{}: {} - {}", i + 1, name, desc))
        .collect::<Vec<_>>()
        .join("\n");
    
    format!(
        r#"You are organizing clusters of "{}" into higher-level categories. Below are {} existing clusters:

{}

Create {} higher-level categories that group these clusters thematically. Each category should:
1. Have a clear, descriptive name (3-10 words)
2. Have a brief description explaining the theme (1-2 sentences)
3. Be broad enough to contain multiple clusters but specific enough to be meaningful

Format your response as a numbered list:
1. Name: [category name]
   Description: [category description]
2. Name: [category name]
   Description: [category description]
..."#,
        facet.name,
        cluster_summaries.len(),
        clusters_text,
        n_desired_names
    )
}

pub fn get_deduplicate_cluster_names_prompt(
    category_names: &[(String, String)],
) -> String {
    let names_text = category_names
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| format!("{}: {} - {}", i + 1, name, desc))
        .collect::<Vec<_>>()
        .join("\n");
    
    format!(
        r#"Review these category names and descriptions for duplicates or near-duplicates:

{}

Identify which categories are essentially the same and should be merged. For each group of duplicates, provide:
1. The indices of the duplicate categories (comma-separated)
2. A single merged name and description

Format your response as:
Merge [indices]: Name: [merged name], Description: [merged description]

If there are no duplicates, respond with: "No duplicates found""#,
        names_text
    )
}

pub fn get_assign_to_high_level_cluster_prompt(
    lower_level_clusters: &[(String, String)],
    higher_level_categories: &[(String, String)],
) -> String {
    let clusters_text = lower_level_clusters
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| format!("{}: {} - {}", i + 1, name, desc))
        .collect::<Vec<_>>()
        .join("\n");
    
    let categories_text = higher_level_categories
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| format!("{}: {} - {}", char::from(b'A' + i as u8), name, desc))
        .collect::<Vec<_>>()
        .join("\n");
    
    format!(
        r#"Assign each cluster to the most appropriate higher-level category.

Clusters:
{}

Categories:
{}

For each cluster, provide its assignment in the format:
1 -> A
2 -> B
...

Every cluster must be assigned to exactly one category."#,
        clusters_text,
        categories_text
    )
}

pub fn get_renaming_higher_level_cluster_prompt(
    category_name: &str,
    category_description: &str,
    children: &[(String, String)],
) -> String {
    let children_text = children
        .iter()
        .map(|(name, desc)| format!("- {} : {}", name, desc))
        .collect::<Vec<_>>()
        .join("\n");
    
    format!(
        r#"Review this category and its contents, then provide an improved name and description if needed.

Current Category:
Name: {}
Description: {}

Contains these clusters:
{}

Based on what's actually in this category, provide:
1. A name that accurately reflects the contents (3-10 words)
2. A description that captures what unites these clusters (1-2 sentences)

Format your response as:
Name: [improved name]
Description: [improved description]"#,
        category_name,
        category_description,
        children_text
    )
}

pub fn get_summarize_facet_prompt(data: &str) -> String {
    format!(
        r#"Summarize the following data concisely:

{}

Provide a clear, informative summary in 1-3 sentences."#,
        data
    )
}