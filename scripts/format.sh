#!/usr/bin/env bash
set -x

ruff check stanza tests scripts --fix
ruff format stanza tests scripts
