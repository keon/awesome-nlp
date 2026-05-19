# Atlas Cloud Provider Review

## Summary

This repository is an awesome list, not a runnable MAAS application. There is no provider registry, adapter factory, SDK wrapper, server route, or environment configuration to extend in-repo.

To keep the change minimal and PR-friendly, the integration for this repository is implemented as:

- a README service entry for Atlas Cloud
- a tracked logo asset used by that entry
- local-only integration validation against the Atlas Cloud OpenAI-compatible API

## What Changed

### README

- Added `Atlas Cloud` to the `Services` section.
- Used the required tracked link format:
  `https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=awesome-nlp`
- Kept the description focused on NLP-relevant capabilities:
  translation, summarization, multilingual generation, and structured extraction.
- Added an inline logo to satisfy the requested branding change while keeping the diff small.

### Assets

- Added `assets/atlas-cloud.png` from the provided local logo file.

## Provider Plan For This Repo

Because this repository has no runtime provider system, the practical provider plan is:

1. expose Atlas Cloud as a documented NLP service in `README.md`
2. validate the OpenAI-compatible chat endpoint locally
3. avoid introducing non-existent adapter code that would not be used anywhere in this repo

## Atlas Cloud Paths Used

These paths come from the Atlas Cloud documentation:

- LLM base URL: `https://api.atlascloud.ai/v1`
- Chat completions: `POST /v1/chat/completions`
- Image generation: `POST /api/v1/model/generateImage`
- Video generation: `POST /api/v1/model/generateVideo`
- Media upload: `POST /api/v1/model/uploadMedia`
- Prediction polling: `GET /api/v1/model/prediction/{id}`

## Local Validation Scope

The local integration validation is intentionally not committed to this repository.

Validation targets:

- non-streaming chat completion
- streaming chat completion
- OpenAI-compatible request shape
- authentication through a local-only API key file outside the repo

## Local Validation Result

Validated successfully outside the repo with the provided local API key.

- `GET https://api.atlascloud.ai/v1/models` returned the available model catalog.
- `POST https://api.atlascloud.ai/v1/chat/completions` succeeded with a real model ID:
  `deepseek-ai/deepseek-v4-flash`
- streaming SSE also succeeded and returned the expected token stream.
- OpenAI Python SDK validation also succeeded with:
  `base_url="https://api.atlascloud.ai/v1"`

## Important Compatibility Note

Atlas Cloud's public docs show shorthand examples such as `deepseek-v3`, but local validation with this API key required using actual catalog IDs from `/v1/models`.

Working example used for validation:

- `deepseek-ai/deepseek-v4-flash`

## Notes For PR Review

- This is the smallest meaningful change set for `awesome-nlp`.
- A deeper provider implementation would require a different target repository that actually contains MAAS runtime code.
