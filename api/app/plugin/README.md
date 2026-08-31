# AgentSSOT MCP plugin

This directory is the repository source for the AgentSSOT MCP integration. It
contains the server, lightweight session hooks, and the `/hive` skill.

The default `core` profile exposes only bounded memory operations. Set
`HIVE_MCP_PROFILE=operator` only in a separately controlled operator
installation to expose administrative, lifecycle, Cortex, synthesis, loadout,
and Synapse tools.

The client reads its endpoint, key, and default namespace from
`HIVE_AGENT_JSON`. Deployments should install or package this directory through
their agent runtime rather than maintaining an untracked second source copy.
