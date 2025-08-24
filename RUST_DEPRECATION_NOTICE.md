# Rust Implementation Deprecation Notice

## ⚠️ Important: Rust Implementation Deprecated

**Effective Date**: Current  
**Migration Deadline**: Immediate

## Status

The Rust implementation of BriefX has been **deprecated** and is **no longer maintained**. 

## Recommended Action

**Please use the Python implementation located in the `python/` directory.**

## Migration Guide

### From Rust to Python

1. **Stop using Rust binaries**:
   ```bash
   # Don't use these anymore
   ./target/release/briefx
   cargo run
   ```

2. **Switch to Python implementation**:
   ```bash
   # Use these instead
   cd python/
   python app.py
   python cli_simple.py
   ```

3. **Update your workflows**:
   - Replace `cargo build` with `pip install -r requirements.txt`
   - Replace `./target/release/briefx ui` with `python app.py`
   - Replace `cargo test` with `python tests/test_updated.py`

## Why This Change?

- **Faster Development**: Python enables quicker iteration and feature development
- **Better Ecosystem**: Access to rich Python data science and ML libraries
- **Easier Deployment**: Simplified deployment without compilation requirements
- **Community Support**: Larger Python community for contributions and support

## What's Available in Python

The Python implementation includes all core features:

- ✅ Conversation analysis and clustering
- ✅ Facet extraction with multiple LLM providers  
- ✅ Web interface and REST API
- ✅ Command line tools
- ✅ Data preprocessing and validation
- ✅ Export capabilities
- ✅ Comprehensive testing

## Support

- **Python Implementation**: Full support and active development
- **Rust Implementation**: No support, bug fixes, or new features

## Questions?

If you have questions about the migration:

1. Check the [Python documentation](python/README.md)
2. Run `python cli_simple.py --help` for CLI options
3. Review [examples](python/briefx/examples.py)
4. Open an issue on GitHub

---

**Thank you for your understanding. The Python implementation provides a better development experience with the same powerful conversation analysis capabilities.**