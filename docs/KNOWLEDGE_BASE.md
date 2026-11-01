# Add your own reference documents

The agent has a local reference knowledge base. You can add manuals, team coding
standards, architecture notes, API descriptions, and worked examples. Retrieval
adds relevant passages to the model's context; it does not retrain the model.
No URLs are fetched, no embedding model is downloaded, and indexing never calls
Ollama or any network service. qwen2.5-coder:7b remains the only model.

## Add documents

Create a knowledge folder inside the project you are working on:

```sh
mkdir -p /path/to/project/knowledge
```

Copy your own `.md`, `.txt`, or `.rst` documents into that folder. For example:

```text
my-project/
  knowledge/
    langgraph/
      state-graph.md
      conditional-edges.md
    team-style-guide.md
    architecture.txt
```

Then ask a coding question:

```sh
offline-agent --workspace /path/to/project \
  'Use my references to explain how to route a LangGraph action'
```

Startup tells you where to add references and how many documents were indexed.
The task automatically retrieves relevant passages. The model can request more
using the `retrieve` tool, with its question in `content` and an empty `path`.
Results include source names and line ranges, such as
`user/team-style-guide.md:12-25`. The model is instructed to cite those sources;
review its citations because generation can still make mistakes.

## Use a separate reference library

```sh
offline-agent --workspace /path/to/project \
  --knowledge-base /path/to/my-reference-library \
  'Consult my architecture references before proposing a change'
```

The supplied directory must exist. Documents can be outside the project, but only
the user-configured reference root is indexed. The model cannot select a different
root. Knowledge retrieval reads documents; it never edits that reference library.
If references are inside the workspace, existing file tools can still edit them
with the usual write approval.

## LangGraph manuals

A short, project-authored LangGraph reference ships with the package and is always
included. It is not the full official manual. To add a full manual, place text or
Markdown copies that you already have permission to use in your local knowledge
folder. Split long manuals into smaller files (one topic/page per file), each no
larger than 64 KB. Include the manual version and source URL in the document text
for provenance. URLs remain text; the agent never follows them. There is no
website downloader or automatic documentation updater.

PDF, Word, HTML, and images are not supported by this first version. Export their
useful content to Markdown or UTF-8 plain text before adding it. Only add documents
you intend to expose to your local coding model. In this repository `knowledge/`
is ignored by Git; add the same rule to other projects if references should remain
unpublished.

## How local RAG works

At startup, the agent reads eligible reference files and builds an in-memory BM25
keyword index. It breaks documents into passages of up to 30 lines / about 1,800
characters and ranks them by query term overlap and rarity. It returns up to four
passages with citations and a 16 KB JSON cap. There are no embeddings or semantic
vector search, so queries using the document's actual terminology work best.

The index is rebuilt for each new task, so edits and deletions are picked up on
the next run. No database, persistent index, account, or API key is required.
Indexing considers at most 1,000 eligible files per root, 200 documents total,
5 MB of document text, and 4,000 passages. Hidden/excluded paths, symlinks, binary
files, invalid UTF-8, and files over 64 KB are excluded. Metadata flags report
index/result limits; a no-match result is not proof the full library has no answer.
The skipped-document count covers read failures among eligible text candidates,
not every excluded path.

References are untrusted data. The model is instructed to use them as evidence,
not follow embedded commands. This is guidance rather than a prompt-injection
security boundary. Existing write/test approvals still apply. Project test code
retains the previously documented user/network permissions; RAG itself performs
no network requests.
