# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Briefcase AI design system applied across all UI components (navbar, buttons, cards, modals)
- Dark navy navbar with pill-shaped active tab indicators
- Blue gradient CTA buttons matching briefcasebrain.com brand palette
- CSS custom properties (`--bc-*`) for consistent theming across all stylesheets

### Changed
- Updated all documentation to reflect Python-only implementation
- Rewrote CONTRIBUTING.md, development, architecture, configuration, and quickstart guides for Python
- Removed stale Rust-specific content from docs

## [0.3.0] - 2025-01-22

### Added
- Python implementation promoted to sole active implementation
- Briefcase AI brand design system (`--bc-*` CSS variables)
- Briefcase AI platform integration via `BRIEFCASE_API_KEY`

### Changed
- Rust implementation fully deprecated; Python is the only maintained version
- Improved error recovery with exponential backoff
- Enhanced monitoring and health check endpoints

## [0.2.1] - 2025-01-21

### Added
- Comprehensive API key setup helper with provider-specific instructions
- Connection testing with real-time feedback and visual status indicators
- JSON, CSV, and HTML export functionality
- Three analysis presets: Quick (5 min), Standard (15 min), Deep (30+ min)
- WebSocket integration for real-time progress updates
- Interactive onboarding flow with guided setup

### Fixed
- Analysis endpoint now applies default config values
- Export buttons connected to working backend implementations
- Progress tracking via WebSocket

## [0.2.0] - 2025-01-21

### Added
- Complete Clio methodology implementation based on the research paper
- Privacy-preserving conversation analysis with hierarchical clustering
- Interactive D3.js force-directed graph visualizations
- Progressive disclosure UI with expertise-based adaptation
- 5-step onboarding flow for new users
- Dynamic data loader supporting WebSocket/SSE/polling
- Real-time dashboard with live metrics and insights
- Advanced Clio UI with Map View, Patterns, Timeline, and Audit tabs
- Google Gemini LLM provider support
- HuggingFace provider support for local embeddings
- MIT license

### Fixed
- Navbar functionality with working tab switching
- Proper HTML file serving from Flask
- Compilation errors in integration tests

### Changed
- Modernized UI with card-based design and smooth transitions
- Enhanced error recovery with automatic retry mechanisms
- Improved logging and monitoring capabilities

## [0.1.0] - 2025-01-20

### Added
- Initial Python implementation
- Core conversation analysis functionality
- Flask web interface with REST API
- Multiple LLM provider support (OpenAI, Anthropic, Ollama)
- Clustering and UMAP visualization
- Session management with pause/resume capability
- Smart preprocessing and validation
- Privacy-focused PII detection and anonymization
- SQLite persistence with PostgreSQL support
