#!/usr/bin/env bash
set -x

ruff check operator_ tests scripts --fix
ruff format operator_ tests scripts