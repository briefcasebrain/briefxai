# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2025-01-21
- Proper error handling with typed errors (`src/error.rs`)
- CI/CD pipeline configuration with GitHub Actions
- Code formatting configuration (`rustfmt.toml`, `clippy.toml`)
- Comprehensive test utilities and fixtures (`tests/common/`)
- Data validation with file size and format checks
- Interactive onboarding flow with guided setup
- Results dashboard with executive summary view
- Context-aware error recovery with suggested actions

### Changed - 2025-01-21
- Professionalized repository documentation (removed excessive emojis)
- Improved code quality with clippy linting fixes
- Enhanced privacy filter with proper statistics tracking
- Updated interactive map to include active overlay tracking
- Refactored test assertions for better reliability
- Improved enum implementations using derive macros

### Fixed - 2025-01-21
- Privacy filter now properly tracks merged and filtered clusters
- Interactive map export includes points array with coordinates
- Test failures in `test_privacy_filtering_integration` and `test_interactive_map_creation`
- Clippy warnings for derivable implementations and code patterns
- Trailing whitespace issues in source files
- Unused import warnings in web module

## [0.2.1] - 2025-01-21 - User Journey Improvements Implemented

### Improvements Completed (2025-01-21)

#### API Key Management
- Added comprehensive API key setup helper with provider-specific instructions
- Implemented connection testing with real-time feedback
- Added visual status indicators for API connectivity
- Included direct links to API key generation pages

#### Export Functionality
- Implemented JSON export with metadata and timestamps
- Added CSV export for cluster data with proper escaping
- Created HTML report generation with formatted output
- All export functions now fully operational

#### Configuration Presets
- Added three preset modes: Quick (5min), Standard (15min), Deep (30+min)
- Each preset optimizes model selection, batch size, and processing depth
- Visual feedback shows selected preset
- Settings automatically adjust based on preset choice

#### WebSocket Integration
- WebSocket handler already implemented in backend
- Client-side connection code functional
- Real-time progress updates ready for use

### Previous State Assessment

#### Functional Components
- Web server running successfully on port 8080
- File upload API endpoint (`/api/upload`) processing JSON files
- Example data generation (`/api/example`) working
- Static file serving for HTML/CSS/JS assets
- Basic UI navigation between screens

#### Partially Functional
- **Analysis endpoint** - Requires complete config object (missing defaults)
- **Progress tracking** - UI exists but lacks WebSocket integration
- **File validation** - Upload works but missing real-time feedback
- **Results display** - Shows data but export functions not connected

#### Non-Functional
- **WebSocket/SSE** - No real-time updates during analysis
- **Clio features** - UI present but backend integration incomplete
- **Session persistence** - Cannot save/resume analysis sessions
- **Export functionality** - Buttons exist without implementation

### Critical User Journey Issues

1. **Onboarding Friction**
   - Users unclear on expected data format
   - No sample data preview or format documentation
   - Missing validation feedback

2. **Configuration Overwhelm**
   - All parameters exposed without context
   - No smart defaults based on data size
   - Missing presets for common use cases

3. **Analysis Black Box**
   - No progress updates during processing
   - Missing time estimates
   - No stage-by-stage feedback

4. **Results Confusion**
   - Overwhelming data presentation
   - No guided tour or key insights
   - Export options non-functional

### Improvement Roadmap

#### Immediate Fixes Required
1. Add default config values to `/api/analyze`
2. Implement WebSocket for real-time updates
3. Connect export functionality
4. Add comprehensive error handling

#### Next Phase Enhancements
1. Interactive onboarding with data preview
2. Config presets (Quick/Standard/Deep)
3. Real-time validation feedback
4. Results tour with key insights first

## [0.2.0] - 2025-01-21

### Added
- Complete Clio methodology implementation from research paper
- Privacy-preserving conversation analysis with hierarchical clustering
- Interactive D3.js force-directed graph visualizations
- Progressive disclosure UI with expertise-based adaptation
- Comprehensive 5-step onboarding flow for new users
- Dynamic data loader supporting WebSocket/SSE/polling
- Real-time dashboard with live metrics and insights
- Advanced Clio UI with Map View, Patterns, Timeline, and Audit tabs
- Gemini LLM provider support
- MIT license file

### Fixed
- Axum router configuration for proper HTML file serving
- Navbar functionality with working tab switching
- InvestigationQuery struct missing sort_criterion field
- Command documentation correction (serve → ui)
- Compilation errors in integration tests

### Changed
- Modernized UI with card-based design and smooth transitions
- Enhanced error recovery with automatic retry mechanisms
- Improved logging and monitoring capabilities
- Updated README with complete feature documentation
- Added comprehensive CONTRIBUTING.md guidelines

## [0.1.0] - 2025-01-20

### Added
- Initial release of BriefXAI
- Core conversation analysis functionality
- Web-based user interface
- Multiple LLM provider support (OpenAI, Ollama)
- Clustering and visualization features
- Session management with pause/resume capability
- Smart preprocessing and validation
- Privacy-focused features with PII detection

### Changed
- Migrated from Python to Rust for improved performance
- Enhanced architecture for production readiness

### Security
- Added PII detection and data validation
- Implemented secure session management