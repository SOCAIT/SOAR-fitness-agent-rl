# 🤝 Contributing to Fitness Reasoning RL Agent

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on code, not personal criticism
- Help others learn and grow

## 🚀 Getting Started

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/fitness-reasoning-rl-agent.git
cd fitness-reasoning-rl-agent
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-description
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with dev tools
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## 💻 Development Workflow

### Code Style

We follow **PEP 8** with some customizations:

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/ --fix

# Type checking with mypy
mypy src/
```

### Commit Messages

Follow conventional commits:

```
feat: add recipe caching to improve search speed
fix: resolve macro calculation rounding error
docs: update README with new examples
test: add unit tests for nutrition validator
refactor: simplify agent tool registration
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_rewards.py

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_rewards.py::test_macro_validation
```

## 🎯 Types of Contributions

### Bug Reports

Found a bug? Create an issue with:

1. **Clear title**: "Agent times out on large scenarios"
2. **Description**: What happened vs. what you expected
3. **Reproduction steps**: Exact steps to reproduce
4. **Environment**: OS, Python version, CUDA version
5. **Logs**: Error messages and traceback
6. **Screenshots**: If applicable

### Feature Requests

Want a new feature? Describe:

1. **Use case**: Why do you need this?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Examples**: How would users use it?

### Code Improvements

Contributing code? Follow these steps:

#### 1. Small Changes (Bug Fixes, Documentation)

```bash
# Make your changes
git add .
git commit -m "fix: resolve issue with macro calculation"
git push origin bugfix/macro-fix

# Create Pull Request on GitHub
```

#### 2. Medium Changes (New Features, Refactoring)

```bash
# Create feature branch
git checkout -b feature/recipe-caching

# Make changes
# Test thoroughly
pytest -v

# Format code
black .
ruff check . --fix
mypy .

# Commit with clear message
git commit -m "feat: add recipe caching to improve search speed"

# Push and create PR
git push origin feature/recipe-caching
```

#### 3. Large Changes (Major Refactoring, New Subsystems)

- **Discuss first**: Create an issue to discuss the approach
- **Follow guidelines**: Ensure alignment with project goals
- **Break into smaller PRs**: Split large changes when possible
- **Document thoroughly**: Add docstrings and comments
- **Add tests**: Aim for >80% test coverage

## 🧪 Testing Guidelines

### Writing Tests

```python
# tests/test_rewards.py
import pytest
from src.env.verifiable_rewards.nutrition_rewards import verify_daily_meal_plan_macros

def test_macro_validation_within_tolerance():
    """Test that macros within ±5% pass validation."""
    meal_plan = {
        "meals": [
            {
                "name": "Chicken & Rice",
                "calories": 700,
                "proteins": 50,
                "carbs": 50,
                "fats": 20,
                "sequence": 1
            }
        ]
    }
    
    score, info = verify_daily_meal_plan_macros(
        meal_plan,
        daily_cal_target=700,
        daily_prot_target=50
    )
    
    assert score >= 0.95  # Should be high for exact match

def test_macro_validation_outside_tolerance():
    """Test that macros outside ±5% fail validation."""
    meal_plan = {
        "meals": [
            {
                "name": "Chicken & Rice",
                "calories": 1000,
                "proteins": 20,
                "carbs": 50,
                "fats": 20,
                "sequence": 1
            }
        ]
    }
    
    score, info = verify_daily_meal_plan_macros(
        meal_plan,
        daily_cal_target=700,
        daily_prot_target=50
    )
    
    assert score < 0.5  # Should be low for mismatch
```

### Test Organization

```
tests/
├── __init__.py
├── test_rewards.py          # Reward calculation tests
├── test_agent.py            # Agent behavior tests
├── test_tools.py            # Tool functionality tests
├── test_data_loader.py      # Data loading tests
└── conftest.py              # Shared fixtures
```

## 📝 Documentation

### Docstrings

Use Google-style docstrings:

```python
def verify_macro_accuracy(meal_plan: dict, targets: dict) -> tuple[float, str]:
    """Verify meal plan macros match targets within tolerance.
    
    Args:
        meal_plan: Dictionary with meal data including macros
        targets: Target nutrition data (calories, protein, etc.)
    
    Returns:
        Tuple of (score, explanation) where score is 0-1
    
    Raises:
        ValueError: If meal_plan is invalid
    
    Examples:
        >>> plan = {"meals": [{"calories": 700, "proteins": 50}]}
        >>> score, info = verify_macro_accuracy(plan, {"daily_cal_target": 700})
        >>> assert score > 0.95
    """
```

### README Updates

When adding features, update relevant sections:

- **Features**: Add to feature list
- **Usage**: Add example usage
- **Configuration**: Document new config options
- **Architecture**: Update diagrams if applicable

## 🔄 Pull Request Process

### Before Submitting

1. **Sync with main:**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run tests:**
   ```bash
   pytest -v
   ```

3. **Check code quality:**
   ```bash
   black .
   ruff check .
   mypy .
   ```

4. **Update documentation:**
   - README.md
   - Docstrings
   - Comments where needed

### Submitting PR

1. Push your branch:
   ```bash
   git push origin feature/your-feature
   ```

2. Create Pull Request with:
   - **Title**: Clear, concise description
   - **Description**: What does this PR do?
   - **Why**: Why is this change needed?
   - **Testing**: How was this tested?
   - **Screenshots**: If applicable (UI changes, diagrams)
   - **Checklist**: ✓ Tests pass, ✓ Docs updated, etc.

3. **PR Template:**
   ```markdown
   ## Description
   Brief explanation of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring
   
   ## Related Issue
   Fixes #123
   
   ## Testing
   - [ ] Added unit tests
   - [ ] Tested locally
   - [ ] All tests pass
   
   ## Documentation
   - [ ] Updated README
   - [ ] Updated docstrings
   - [ ] Added comments for complex logic
   ```

### Review Process

- At least one maintainer will review
- Address feedback and update PR
- Rebase to keep history clean
- Approved PRs will be merged

## 🎓 Development Tips

### Debugging Agent Issues

```python
# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)

# Test tools individually
from src.agent.tools import create_agent_tools
tools = create_agent_tools(...)
result = tools[0].func("test query")
```

### Testing Rewards Offline

```bash
# Run main.py to test reward calculation
python main.py

# Output shows reward scores for sample scenarios
```

### Profiling Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## 📚 Resources

- [OpenPipe ART Documentation](https://docs.openpipe.ai/art)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Pinecone API Reference](https://docs.pinecone.io/)
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## 🎖️ Recognition

Contributors are recognized in:
- Pull request merges
- CHANGELOG.md
- GitHub contributors page
- Project announcements

## ❓ Questions?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Open an issue with `[question]` tag
- Reach out to maintainers

---

Thank you for contributing! 🙏

