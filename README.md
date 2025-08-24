# BriefX - Intelligent Conversation Analysis Platform

BriefX is a powerful conversation analysis platform that helps you understand patterns, extract insights, and visualize relationships in conversational data. Whether you're analyzing customer support tickets, research interviews, or team communications, BriefX provides the tools you need to make sense of your conversations.

## 🚀 Try It Now

**[Launch BriefX](https://briefx-416764050725.us-central1.run.app)** - No installation required!

## ✨ Features

### Core Capabilities
- **Smart Clustering** - Automatically groups similar conversations to identify patterns and themes
- **Facet Extraction** - Extracts key attributes like sentiment, intent, urgency, and complexity
- **Interactive Visualizations** - Explore your data through dynamic charts and graphs
- **Topic Analysis** - Identifies main topics and themes across conversations
- **Privacy-First Design** - Your data stays secure with configurable privacy thresholds

### Analysis Options
- **Free Demo Mode** - Get started immediately with rule-based analysis, no API keys required
- **Premium Providers** - Connect your own OpenAI, Anthropic, or Google Gemini API keys for advanced AI-powered analysis
- **Flexible Processing** - Support for JSON, CSV, and plain text conversation formats

## 🎯 Use Cases

- **Customer Support** - Analyze support tickets to identify common issues and improve response strategies
- **User Research** - Extract insights from user interviews and feedback sessions
- **Team Analytics** - Understand communication patterns within your organization
- **Content Analysis** - Process and categorize large volumes of conversational content
- **Market Research** - Analyze customer conversations to identify trends and opportunities

## 🛠️ Technology Stack

- **Backend**: Python with Flask framework
- **Analysis Engine**: Scikit-learn for clustering, custom NLP pipelines
- **Frontend**: Pure JavaScript with D3.js and Chart.js for visualizations
- **Deployment**: Google Cloud Run for scalable, serverless hosting
- **AI Providers**: Support for OpenAI, Anthropic, and Google Gemini APIs

## 📊 How It Works

1. **Upload Your Data** - Import conversations in JSON, CSV, or text format
2. **Configure Analysis** - Choose between free demo mode or connect your API keys
3. **Process & Analyze** - The platform clusters conversations and extracts insights
4. **Explore Results** - Interactive dashboards help you understand patterns and trends
5. **Export Findings** - Download your analysis results for further use

## 🔧 Local Development

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/briefx.git
cd briefx
```

2. Install dependencies:
```bash
cd python
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Add your API keys if you want to use premium providers
```

4. Run the application:
```bash
python app.py
```

5. Open your browser to `http://localhost:8080`

### Docker Deployment

Build and run with Docker:
```bash
docker build -t briefx .
docker run -p 8080:8080 briefx
```

## 🔑 API Configuration

BriefX works out of the box with the free demo mode. To unlock advanced features, you can configure API keys for:

- **OpenAI** - For GPT-based analysis
- **Anthropic** - For Claude-based analysis
- **Google Gemini** - For Gemini-based analysis

API keys can be configured through the web interface or environment variables.

## 📝 Data Formats

### JSON Format
```json
{
  "conversations": [
    {
      "id": "conv1",
      "messages": [
        {"role": "user", "content": "Hello, I need help"},
        {"role": "assistant", "content": "How can I assist you?"}
      ]
    }
  ]
}
```

### CSV Format
```csv
conversation_id,timestamp,speaker,message
conv1,2024-01-01 10:00,user,"Hello, I need help"
conv1,2024-01-01 10:01,assistant,"How can I assist you?"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with modern web technologies and designed for scalability and ease of use. Special thanks to the open-source community for the amazing tools that make this project possible.

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub or contact the maintainers.

---

**[Try BriefX Now →](https://briefx-416764050725.us-central1.run.app)**