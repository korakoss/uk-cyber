This is a project that aims to analyze a UK cybercrime dataset with the aim of making a good synthetic estimate of total annual cybercrime data.

You may be asked to perform various tasks, such as familiarizing yourself with the dataset, writing a new piece of code, interpreting run results, investigating a question on your own, or brainstorming about approaches interactively. 


# Rules

At the start of every conversation, read NOTES.md before doing anything else. At the end of every conversation turn where new context has accumulated (decisions made, findings, plans updated), update NOTES.md to reflect it. Keep NOTES.md as the authoritative running record of project context, methodology decisions, open questions, and current status — written so that a future session can pick up the thread without needing to re-derive anything from the conversation history.

All data analyses must be written as Python files and run from those files — never as inline bash one-liners. This preserves a log of what was looked at. Policy for analysis files:
- Do not overwrite or delete existing analysis files
- Append to an existing file if adding to the same topic
- Open a new file for a new topic
- Analysis scripts live in src/estimation/ (or src/ for earlier exploratory work)


# Guidelines

Have awareness about what mode the conversation is in – whether the current task is to implement a well-defined narrow task or more like brainstorming approaches, thinking about the problem. 

Think of the workflow as alternating cycles of breadth-first exploration/orientation and thinking first, then once we collected enough information to determine a narrow, crisp "direction"/experiment to try, implementing that in a more focused, depth-first mode. Do not prematurely go too deep in a direction without sufficient exploratory orientation first.

In the former brainstorming/idea generation mode, generally prefer reasoning from first principles with lighter conceptual machinery; only moving on to more involved conceptual tools when the low-hanging fruit was picked with the former approach.

Do not write wall-of-text outputs to the user unless asked to report in detail. Aim for an "executive summary" vibe instead, such that the user is able to efficiently make informed decisions about the direction to take, without getting overwhelmed by too much information. 

Retain awareness of what the current high-level objective is. Ask the user to spell it out if unclear, rather than getting lost in the weeds and losing sight of what the current goal is.
