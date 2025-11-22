# Changelog

All notable changes to the Fitness Reasoning RL Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-16

### Added

- **Initial Release** 🎉
  - Core RL training framework using OpenPipe ART
  - LangGraph-based ReAct agent for meal planning
  - Verifiable rewards system (no LLM judge needed)
  - Nutrition macro validation (±5% tolerance)
  - JSON schema validation for meal plans
  - Pinecone integration for recipe search
  - Weights & Biases integration for experiment tracking
  - Support for dietary restrictions and preferences

- **Features**
  - Personalized daily meal plan generation
  - Macro-accurate nutrition planning
  - RAG-based recipe retrieval
  - GRPO (Group Relative Policy Optimization) training
  - Configuration system for flexible parameter tuning
  - Docker-ready training setup
  - Shell script for easy cloud deployment

- **Documentation**
  - Comprehensive README with architecture diagrams
  - Installation guides (quick start & manual)
  - Usage examples and API reference
  - Contributing guidelines
  - Troubleshooting guide
  - Mermaid diagrams for system visualization

- **Development Tools**
  - Pre-configured linting (Black, Ruff, mypy)
  - Test setup with pytest
  - Development dependencies
  - Git hooks configuration
  - PyProject.toml for modern Python packaging

- **Example Data**
  - Synthetic fitness scenarios generator
  - Joint exercises metadata
  - Sample configuration files

### Technical Details

- **Base Model**: Qwen2.5-7B-Instruct
- **Framework**: OpenPipe ART + LangGraph
- **Vector DB**: Pinecone
- **LLM API**: OpenAI GPT-4
- **Experiment Tracking**: Weights & Biases
- **Python Support**: 3.10+

### Known Limitations

- Currently focused on nutrition planning (workout planning in progress)
- Requires GPU for training (T4 or better recommended)
- Pinecone API key required for recipe retrieval
- Max 30 reasoning steps per agent rollout
- Macro tolerance set to ±5%

---

## Future Roadmap

### [0.2.0] - Planned

- [ ] Workout plan generation
- [ ] Multi-day plan optimization
- [ ] User preference learning
- [ ] Recipe variation suggestions
- [ ] Allergen filtering
- [ ] Cost optimization
- [ ] Sustainability scoring

### [0.3.0] - Planned

- [ ] REST API for inference
- [ ] Web dashboard
- [ ] Real-time plan adjustments
- [ ] Integration with fitness trackers
- [ ] Mobile app
- [ ] Multi-language support

### [0.4.0] - Planned

- [ ] Fine-tuned models for different diets
- [ ] Ensemble agent approaches
- [ ] Transfer learning from nutrition domain
- [ ] Advanced reasoning chains
- [ ] Few-shot learning capabilities

---

## Notes

- For detailed API changes between versions, check individual commit messages
- Breaking changes will be noted prominently in release notes
- Deprecated features will have a 2-release deprecation period

