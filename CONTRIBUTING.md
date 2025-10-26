# Contributing to EyeController

Thank you for your interest in contributing to the EyeController project! We welcome contributions, whether they are bug reports, feature suggestions, code improvements, or documentation updates.

By participating in this project, you agree to abide by the principles set forth here.

## Ways to Contribute

### 1. Reporting Bugs and Issues

If you find a bug or experience unexpected behavior, please check the existing issues before opening a new one.

* **When reporting a bug, please include:**
    * A clear and descriptive title.
    * The steps to reproduce the error reliably.
    * Your Operating System (e.g., Windows 10/11) and Python version (e.g., 3.11).
    * Any relevant output, logs, or screenshots.

### 2. Suggesting Enhancements

If you have an idea for a new feature, a performance improvement, or better configuration, please open an issue with the label **"Enhancement"**.

* Describe the feature and why it would be beneficial to the project.
* If possible, outline a potential implementation approach.

### 3. Code Contributions (Pull Requests)

We welcome pull requests for bug fixes, new features, and improvements to the existing codebase.

#### Development Setup

1.  **Fork** the repository to your own GitHub account.
2.  **Clone** your fork to your local machine.
3.  **Set up the development environment** (ensure you are using a compatible Python version, 3.8 to 3.11):
    ```bash
    uv venv
    .\.venv\Scripts\activate
    uv pip install -r requirements.txt
    ```

#### Pull Request Guidelines

1.  Create a **new branch** for your work: `git checkout -b feature/your-new-feature` or `bugfix/issue-number`.
2.  Ensure your code **adheres to general Python style** and is clear and well-commented.
3.  Keep commits logical and atomic. Write a clear, concise **commit message** summarizing your changes.
4.  If your change requires a new dependency, update the `requirements.txt` file.
5.  **Test your changes** thoroughly before submitting.
6.  Submit a **Pull Request** to the `main` branch of this repository.

#### Pull Request Review

All pull requests will be reviewed by the maintainer. We may request changes or clarification before merging.

## Style and Standards

* **Type Hinting:** Please use Python type hints where appropriate.
* **Docstrings:** Use docstrings for all major functions and classes.
* **Libraries:** Try to limit external dependencies. If a new library is needed, state the reason clearly in your PR.

Thank you for helping to make EyeController better!
