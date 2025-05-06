This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

## Additional Info

# Directory Structure
```
.devcontainer/
  devcontainer.json
  Dockerfile
  init-firewall.sh
.github/
  actions/
    claude-code-action/
      action.yml
    claude-issue-triage-action/
      action.yml
  ISSUE_TEMPLATE/
    bug_report.md
  workflows/
    claude-issue-triage.yml
.gitattributes
CHANGELOG.md
LICENSE.md
README.md
SECURITY.md
```

# Files

## File: .devcontainer/devcontainer.json
````json
{
  "name": "Claude Code Sandbox",
  "build": {
    "dockerfile": "Dockerfile",
    "args": {
      "TZ": "${localEnv:TZ:America/Los_Angeles}"
    }
  },
  "runArgs": [
    "--cap-add=NET_ADMIN",
    "--cap-add=NET_RAW"
  ],
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode",
        "eamodio.gitlens"
      ],
      "settings": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.codeActionsOnSave": {
          "source.fixAll.eslint": "explicit"
        },
        "terminal.integrated.defaultProfile.linux": "zsh",
        "terminal.integrated.profiles.linux": {
          "bash": {
            "path": "bash",
            "icon": "terminal-bash"
          },
          "zsh": {
            "path": "zsh"
          }
        }
      }
    }
  },
  "remoteUser": "node",
  "mounts": [
    "source=claude-code-bashhistory,target=/commandhistory,type=volume",
    "source=claude-code-config,target=/home/node/.claude,type=volume"
  ],
  "remoteEnv": {
    "NODE_OPTIONS": "--max-old-space-size=4096",
    "CLAUDE_CONFIG_DIR": "/home/node/.claude",
    "POWERLEVEL9K_DISABLE_GITSTATUS": "true"
  },
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=delegated",
  "workspaceFolder": "/workspace",
  "postCreateCommand": "sudo /usr/local/bin/init-firewall.sh"
}
````

## File: .devcontainer/Dockerfile
````
FROM node:20

ARG TZ
ENV TZ="$TZ"

# Install basic development tools and iptables/ipset
RUN apt update && apt install -y less \
  git \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq

# Ensure default node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
  chown -R node:node /usr/local/share

ARG USERNAME=node

# Persist bash history.
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace /home/node/.claude && \
  chown -R node:node /workspace /home/node/.claude

WORKDIR /workspace

RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/0.18.2/git-delta_0.18.2_${ARCH}.deb" && \
  sudo dpkg -i "git-delta_0.18.2_${ARCH}.deb" && \
  rm "git-delta_0.18.2_${ARCH}.deb"

# Set up non-root user
USER node

# Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Set the default shell to bash rather than sh
ENV SHELL /bin/zsh

# Default powerline10k theme
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.0/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude
RUN npm install -g @anthropic-ai/claude-code

# Copy and set up firewall script
COPY init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
  chmod 0440 /etc/sudoers.d/node-firewall
USER node
````

## File: .devcontainer/init-firewall.sh
````bash
#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, and pipeline failures
IFS=$'\n\t'       # Stricter word splitting

# Flush existing rules and delete existing ipsets
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
ipset destroy allowed-domains 2>/dev/null || true

# First allow DNS and localhost before any restrictions
# Allow outbound DNS
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
# Allow inbound DNS responses
iptables -A INPUT -p udp --sport 53 -j ACCEPT
# Allow outbound SSH
iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT
# Allow inbound SSH responses
iptables -A INPUT -p tcp --sport 22 -m state --state ESTABLISHED -j ACCEPT
# Allow localhost
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Create ipset with CIDR support
ipset create allowed-domains hash:net

# Fetch GitHub meta information and aggregate + add their IP ranges
echo "Fetching GitHub IP ranges..."
gh_ranges=$(curl -s https://api.github.com/meta)
if [ -z "$gh_ranges" ]; then
    echo "ERROR: Failed to fetch GitHub IP ranges"
    exit 1
fi

if ! echo "$gh_ranges" | jq -e '.web and .api and .git' >/dev/null; then
    echo "ERROR: GitHub API response missing required fields"
    exit 1
fi

echo "Processing GitHub IPs..."
while read -r cidr; do
    if [[ ! "$cidr" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/[0-9]{1,2}$ ]]; then
        echo "ERROR: Invalid CIDR range from GitHub meta: $cidr"
        exit 1
    fi
    echo "Adding GitHub range $cidr"
    ipset add allowed-domains "$cidr"
done < <(echo "$gh_ranges" | jq -r '(.web + .api + .git)[]' | aggregate -q)

# Resolve and add other allowed domains
for domain in \
    "registry.npmjs.org" \
    "api.anthropic.com" \
    "sentry.io" \
    "statsig.anthropic.com" \
    "statsig.com"; do
    echo "Resolving $domain..."
    ips=$(dig +short A "$domain")
    if [ -z "$ips" ]; then
        echo "ERROR: Failed to resolve $domain"
        exit 1
    fi
    
    while read -r ip; do
        if [[ ! "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
            echo "ERROR: Invalid IP from DNS for $domain: $ip"
            exit 1
        fi
        echo "Adding $ip for $domain"
        ipset add allowed-domains "$ip"
    done < <(echo "$ips")
done

# Get host IP from default route
HOST_IP=$(ip route | grep default | cut -d" " -f3)
if [ -z "$HOST_IP" ]; then
    echo "ERROR: Failed to detect host IP"
    exit 1
fi

HOST_NETWORK=$(echo "$HOST_IP" | sed "s/\.[0-9]*$/.0\/24/")
echo "Host network detected as: $HOST_NETWORK"

# Set up remaining iptables rules
iptables -A INPUT -s "$HOST_NETWORK" -j ACCEPT
iptables -A OUTPUT -d "$HOST_NETWORK" -j ACCEPT

# Set default policies to DROP first
# Set default policies to DROP first
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# First allow established connections for already approved traffic
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Then allow only specific outbound traffic to allowed domains
iptables -A OUTPUT -m set --match-set allowed-domains dst -j ACCEPT

echo "Firewall configuration complete"
echo "Verifying firewall rules..."
if curl --connect-timeout 5 https://example.com >/dev/null 2>&1; then
    echo "ERROR: Firewall verification failed - was able to reach https://example.com"
    exit 1
else
    echo "Firewall verification passed - unable to reach https://example.com as expected"
fi

# Verify GitHub API access
if ! curl --connect-timeout 5 https://api.github.com/zen >/dev/null 2>&1; then
    echo "ERROR: Firewall verification failed - unable to reach https://api.github.com"
    exit 1
else
    echo "Firewall verification passed - able to reach https://api.github.com as expected"
fi
````

## File: .github/actions/claude-code-action/action.yml
````yaml
name: "Claude Code Action"
description: "Run Claude Code in GitHub Actions workflows"

inputs:
  github_token:
    description: "GitHub token with repo and issues permissions"
    required: true
  anthropic_api_key:
    description: "Anthropic API key"
    required: true
  prompt:
    description: "The prompt to send to Claude Code"
    required: false
    default: ""
  prompt_file:
    description: "Path to a file containing the prompt to send to Claude Code"
    required: false
    default: ""
  allowed_tools:
    description: "Comma-separated list of allowed tools for Claude Code to use"
    required: false
    default: ""
  output_file:
    description: "File to save Claude Code output to (optional)"
    required: false
    default: ""
  timeout_minutes:
    description: "Timeout in minutes for Claude Code execution"
    required: false
    default: "10"
  install_github_mcp:
    description: "Whether to install the GitHub MCP server"
    required: false
    default: "false"

runs:
  using: "composite"
  steps:
    - name: Install Claude Code
      shell: bash
      run: npm install -g @anthropic-ai/claude-code

    - name: Install GitHub MCP Server
      if: inputs.install_github_mcp == 'true'
      shell: bash
      run: |
        claude mcp add-json github '{
          "command": "docker",
          "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server:sha-ff3036d"
          ],
          "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "${{ inputs.GITHUB_TOKEN }}"
          }
        }'

    - name: Prepare Prompt File
      shell: bash
      id: prepare_prompt
      run: |
        # Check if either prompt or prompt_file is provided
        if [ -z "${{ inputs.prompt }}" ] && [ -z "${{ inputs.prompt_file }}" ]; then
          echo "::error::Neither 'prompt' nor 'prompt_file' was provided. At least one is required."
          exit 1
        fi

        # Determine which prompt source to use
        if [ ! -z "${{ inputs.prompt_file }}" ]; then
          # Check if the prompt file exists
          if [ ! -f "${{ inputs.prompt_file }}" ]; then
            echo "::error::Prompt file '${{ inputs.prompt_file }}' does not exist."
            exit 1
          fi

          # Use the provided prompt file
          PROMPT_PATH="${{ inputs.prompt_file }}"
        else
          mkdir -p /tmp/claude-action
          PROMPT_PATH="/tmp/claude-action/prompt.txt"
          echo "${{ inputs.prompt }}" > "$PROMPT_PATH"
        fi

        # Verify the prompt file is not empty
        if [ ! -s "$PROMPT_PATH" ]; then
          echo "::error::Prompt is empty. Please provide a non-empty prompt."
          exit 1
        fi

        # Save the prompt path for the next step
        echo "PROMPT_PATH=$PROMPT_PATH" >> $GITHUB_ENV

    - name: Run Claude Code
      shell: bash
      id: run_claude
      run: |
        ALLOWED_TOOLS_ARG=""
        if [ ! -z "${{ inputs.allowed_tools }}" ]; then
          ALLOWED_TOOLS_ARG="--allowedTools ${{ inputs.allowed_tools }}"
        fi

        # Set a timeout to ensure the command doesn't run indefinitely
        timeout_seconds=$((${{ inputs.timeout_minutes }} * 60))

        if [ -z "${{ inputs.output_file }}" ]; then
          # Run Claude Code and output to console
          timeout $timeout_seconds claude \
            -p \
            --verbose \
            --output-format stream-json \
            "$(cat ${{ env.PROMPT_PATH }})" \
            ${{ inputs.allowed_tools != '' && format('--allowedTools "{0}"', inputs.allowed_tools) || '' }}
        else
          # Run Claude Code and tee output to console and file
          timeout $timeout_seconds claude \
            -p \
            --verbose \
            --output-format stream-json \
            "$(cat ${{ env.PROMPT_PATH }})" \
            ${{ inputs.allowed_tools != '' && format('--allowedTools "{0}"', inputs.allowed_tools) || '' }} | tee output.txt

          # Process output.txt into JSON in a separate step
          jq -s '.' output.txt > output.json

          # Extract the result from the last item in the array (system message)
          jq -r '.[-1].result' output.json > "${{ inputs.output_file }}"

          echo "Complete output saved to output.json, final response saved to ${{ inputs.output_file }}"
        fi
      env:
        ANTHROPIC_API_KEY: ${{ inputs.anthropic_api_key }}
        GITHUB_TOKEN: ${{ inputs.github_token }}
````

## File: .github/actions/claude-issue-triage-action/action.yml
````yaml
name: "Claude Issue Triage Action"
description: "Automatically triage GitHub issues using Claude Code"

inputs:
  timeout_minutes:
    description: "Timeout in minutes for execution"
    required: false
    default: "5"
  anthropic_api_key:
    description: "Anthropic API key"
    required: true
  github_token:
    description: "GitHub token with repo and issues permissions"
    required: true

runs:
  using: "composite"
  steps:
    - name: Checkout repository code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Create prompt file
      shell: bash
      run: |
        mkdir -p /tmp/claude-prompts
        cat > /tmp/claude-prompts/claude-issue-triage-prompt.txt << 'EOF'
        You're an issue triage assistant for GitHub issues. Your task is to analyze the issue and select appropriate labels from the provided list.

        IMPORTANT: Don't post any comments or messages to the issue. Your only action should be to apply labels.

        Issue Information:
        - REPO: ${{ github.repository }}
        - ISSUE_NUMBER: ${{ github.event.issue.number }}

        TASK OVERVIEW:

        1. First, fetch the list of labels available in this repository by running: `gh label list`. Run exactly this command with nothing else.

        2. Next, use the GitHub tools to get context about the issue:
           - You have access to these tools:
             - mcp__github__get_issue: Use this to retrieve the current issue's details including title, description, and existing labels
             - mcp__github__get_issue_comments: Use this to read any discussion or additional context provided in the comments
             - mcp__github__update_issue: Use this to apply labels to the issue (do not use this for commenting)
             - mcp__github__search_issues: Use this to find similar issues that might provide context for proper categorization and to identify potential duplicate issues
             - mcp__github__list_issues: Use this to understand patterns in how other issues are labeled
           - Start by using mcp__github__get_issue to get the issue details

        3. Analyze the issue content, considering:
           - The issue title and description
           - The type of issue (bug report, feature request, question, etc.)
           - Technical areas mentioned
           - Severity or priority indicators
           - User impact
           - Components affected

        4. Select appropriate labels from the available labels list provided above:
           - Choose labels that accurately reflect the issue's nature
           - Be specific but comprehensive
           - Select priority labels if you can determine urgency (high-priority, med-priority, or low-priority)
           - Consider platform labels (android, ios) if applicable
           - If you find similar issues using mcp__github__search_issues, consider using a "duplicate" label if appropriate. Only do so if the issue is a duplicate of another OPEN issue.

        5. Apply the selected labels:
           - Use mcp__github__update_issue to apply your selected labels
           - DO NOT post any comments explaining your decision
           - DO NOT communicate directly with users
           - If no labels are clearly applicable, do not apply any labels

        IMPORTANT GUIDELINES:
        - Be thorough in your analysis
        - Only select labels from the provided list above
        - DO NOT post any comments to the issue
        - Your ONLY action should be to apply labels using mcp__github__update_issue
        - It's okay to not add any labels if none are clearly applicable
        EOF

    - name: Run Claude Code
      uses: ./.github/actions/claude-code-action
      with:
        prompt_file: /tmp/claude-prompts/claude-issue-triage-prompt.txt
        allowed_tools: "Bash(gh label list),mcp__github__get_issue,mcp__github__get_issue_comments,mcp__github__update_issue,mcp__github__search_issues,mcp__github__list_issues"
        install_github_mcp: "true"
        timeout_minutes: ${{ inputs.timeout_minutes }}
        anthropic_api_key: ${{ inputs.anthropic_api_key }}
        github_token: ${{ inputs.github_token }}
````

## File: .github/ISSUE_TEMPLATE/bug_report.md
````markdown
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Environment
- Platform (select one):
  - [ ] Anthropic API
  - [ ] AWS Bedrock
  - [ ] Google Vertex AI
  - [ ] Other: <!-- specify -->
- Claude CLI version: <!-- output of `claude --version` -->
- Operating System: <!-- e.g. macOS 14.3, Windows 11, Ubuntu 22.04 -->
- Terminal:  <!-- e.g. iTerm2, Terminal App -->

## Bug Description
<!-- A clear and concise description of the bug -->

## Steps to Reproduce
1. <!-- First step -->
2. <!-- Second step -->
3. <!-- And so on... -->

## Expected Behavior
<!-- What you expected to happen -->

## Actual Behavior
<!-- What actually happened -->

## Additional Context
<!-- Add any other context about the problem here, such as screenshots, logs, etc. -->
````

## File: .github/workflows/claude-issue-triage.yml
````yaml
name: Claude Issue Triage
description: "Automatically triage GitHub issues using Claude Code"

on:
  issues:
    types: [opened]

jobs:
  triage-issue:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: read
      issues: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Run Claude Issue Triage
        uses: ./.github/actions/claude-issue-triage-action
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          github_token: ${{ secrets.GITHUB_TOKEN }}
````

## File: .gitattributes
````
* text=auto eol=lf
*.sh text eol=lf
````

## File: CHANGELOG.md
````markdown
# Changelog

## 0.2.100

- Fixed a crash caused by a stack overflow error
- Made db storage optional; missing db support disables --continue and --resume

## 0.2.98

- Fixed an issue where auto-compact was running twice

## 0.2.96

- Claude Code can now also be used with a Claude Max subscription (https://claude.ai/upgrade)

## 0.2.93

- Resume conversations from where you left off from with "claude --continue" and "claude --resume"
- Claude now has access to a Todo list that helps it stay on track and be more organized

## 0.2.82

- Added support for --disallowedTools
- Renamed tools for consistency: LSTool -> LS, View -> Read, etc.

## 0.2.75

- Hit Enter to queue up additional messages while Claude is working
- Drag in or copy/paste image files directly into the prompt
- @-mention files to directly add them to context
- Run one-off MCP servers with `claude --mcp-config <path-to-file>`
- Improved performance for filename auto-complete

## 0.2.74

- Added support for refreshing dynamically generated API keys (via apiKeyHelper), with a 5 minute TTL
- Task tool can now perform writes and run bash commands

## 0.2.72

- Updated spinner to indicate tokens loaded and tool usage

## 0.2.70

- Network commands like curl are now available for Claude to use
- Claude can now run multiple web queries in parallel
- Pressing ESC once immediately interrupts Claude in Auto-accept mode

## 0.2.69

- Fixed UI glitches with improved Select component behavior
- Enhanced terminal output display with better text truncation logic

## 0.2.67

- Shared project permission rules can be saved in .claude/settings.json

## 0.2.66

- Print mode (-p) now supports streaming output via --output-format=stream-json
- Fixed issue where pasting could trigger memory or bash mode unexpectedly

## 0.2.63

- Fixed an issue where MCP tools were loaded twice, which caused tool call errors

## 0.2.61

- Navigate menus with vim-style keys (j/k) or bash/emacs shortcuts (Ctrl+n/p) for faster interaction
- Enhanced image detection for more reliable clipboard paste functionality
- Fixed an issue where ESC key could crash the conversation history selector

## 0.2.59

- Copy+paste images directly into your prompt
- Improved progress indicators for bash and fetch tools
- Bugfixes for non-interactive mode (-p)

## 0.2.54

- Quickly add to Memory by starting your message with '#'
- Press ctrl+r to see full output for long tool results
- Added support for MCP SSE transport

## 0.2.53

- New web fetch tool lets Claude view URLs that you paste in
- Fixed a bug with JPEG detection

## 0.2.50

- New MCP "project" scope now allows you to add MCP servers to .mcp.json files and commit them to your repository

## 0.2.49

- Previous MCP server scopes have been renamed: previous "project" scope is now "local" and "global" scope is now "user"

## 0.2.47

- Press Tab to auto-complete file and folder names
- Press Shift + Tab to toggle auto-accept for file edits
- Automatic conversation compaction for infinite conversation length (toggle with /config)

## 0.2.44

- Ask Claude to make a plan with thinking mode: just say 'think' or 'think harder' or even 'ultrathink'

## 0.2.41

- MCP server startup timeout can now be configured via MCP_TIMEOUT environment variable
- MCP server startup no longer blocks the app from starting up

## 0.2.37

- New /release-notes command lets you view release notes at any time
- `claude config add/remove` commands now accept multiple values separated by commas or spaces

## 0.2.36

- Import MCP servers from Claude Desktop with `claude mcp add-from-claude-desktop`
- Add MCP servers as JSON strings with `claude mcp add-json <n> <json>`

## 0.2.34

- Vim bindings for text input - enable with /vim or /config

## 0.2.32

- Interactive MCP setup wizard: Run "claude mcp add" to add MCP servers with a step-by-step interface
- Fix for some PersistentShell issues

## 0.2.31

- Custom slash commands: Markdown files in .claude/commands/ directories now appear as custom slash commands to insert prompts into your conversation
- MCP debug mode: Run with --mcp-debug flag to get more information about MCP server errors

## 0.2.30

- Added ANSI color theme for better terminal compatibility
- Fixed issue where slash command arguments weren't being sent properly
- (Mac-only) API keys are now stored in macOS Keychain

## 0.2.26

- New /approved-tools command for managing tool permissions
- Word-level diff display for improved code readability
- Fuzzy matching for slash commands

## 0.2.21

- Fuzzy matching for /commands
````

## File: LICENSE.md
````markdown
Claude Code is a Beta research preview per our [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms). If Customer chooses to send us feedback about Claude Code, such as transcripts of Customer Claude Code usage, Anthropic may use that feedback to debug related issues or to improve Claude Code’s functionality (e.g., to reduce the risk of similar bugs occurring in the future). Anthropic will not train models using feedback from Claude Code.

© Anthropic PBC. All rights reserved. Use is subject to Anthropic's [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms).
````

## File: README.md
````markdown
# Claude Code (Research Preview)

![](https://img.shields.io/badge/Node.js-18%2B-brightgreen?style=flat-square) [![npm]](https://www.npmjs.com/package/@anthropic-ai/claude-code)

[npm]: https://img.shields.io/npm/v/@anthropic-ai/claude-code.svg?style=flat-square

Claude Code is an agentic coding tool that lives in your terminal, understands your codebase, and helps you code faster by executing routine tasks, explaining complex code, and handling git workflows - all through natural language commands.

Some of its key capabilities include:

- Edit files and fix bugs across your codebase
- Answer questions about your code's architecture and logic
- Execute and fix tests, lint, and other commands
- Search through git history, resolve merge conflicts, and create commits and PRs

**Learn more in the [official documentation](https://docs.anthropic.com/en/docs/agents/claude-code/introduction)**.

## Get started

1. If you are new to Node.js and Node Package Manager (`npm`), then it is recommended that you configure an NPM prefix for your user.
   Instructions on how to do this can be found [here](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#recommended-create-a-new-user-writable-npm-prefix).

   *Important* We recommend installing this package as a non-privileged user, not as an administrative user like `root`.
   Installing as a non-privileged user helps maintain your system's security and stability.

2. Install Claude Code:
   
   ```sh
   npm install -g @anthropic-ai/claude-code
   ```

3. Navigate to your project directory and run <code>claude</code>.

4. Complete the one-time OAuth process with your Anthropic Console account.

### Research Preview

We're launching Claude Code as a beta product in research preview to learn directly from developers about their experiences collaborating with AI agents. Our aim is to learn more about how developers prefer to collaborate with AI tools, which development workflows benefit most from working with the agent, and how we can make the agent experience more intuitive.

This is an early version of the product experience, and it's likely to evolve as we learn more about developer preferences. Claude Code is an early look into what's possible with agentic coding, and we know there are areas to improve. We plan to enhance tool execution reliability, support for long-running commands, terminal rendering, and Claude's self-knowledge of its capabilities -- as well as many other product experiences -- over the coming weeks.

### Reporting Bugs

We welcome feedback during this beta period. Use the `/bug` command to report issues directly within Claude Code, or file a [GitHub issue](https://github.com/anthropics/claude-code/issues).

### Data collection, usage, and retention

When you use Claude Code, we collect feedback, which includes usage data (such as code acceptance or rejections), associated conversation data, and user feedback submitted via the `/bug` command.

#### How we use your data

We may use feedback to improve our products and services, but we will not train generative models using your feedback from Claude Code. Given their potentially sensitive nature, we store user feedback transcripts for only 30 days.

If you choose to send us feedback about Claude Code, such as transcripts of your usage, Anthropic may use that feedback to debug related issues and improve Claude Code's functionality (e.g., to reduce the risk of similar bugs occurring in the future).

### Privacy safeguards

We have implemented several safeguards to protect your data, including limited retention periods for sensitive information, restricted access to user session data, and clear policies against using feedback for model training.

For full details, please review our [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms) and [Privacy Policy](https://www.anthropic.com/legal/privacy).
````

## File: SECURITY.md
````markdown
# Security Policy
Thank you for helping us keep Claude Code secure!

## Reporting Security Issues

The security of our systems and user data is Anthropic's top priority. We appreciate the work of security researchers acting in good faith in identifying and reporting potential vulnerabilities.

Our security program is managed on HackerOne and we ask that any validated vulnerability in this functionality be reported through their [submission form](https://hackerone.com/anthropic-vdp/reports/new?type=team&report_type=vulnerability).

## Vulnerability Disclosure Program

Our Vulnerability Program Guidelines are defined on our [HackerOne program page](https://hackerone.com/anthropic-vdp).
````
