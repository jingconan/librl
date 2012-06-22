#!/usr/bin/env sh
rm -r ./res/*
find . -iname "*.pyc" -exec rm '{}' ';'
find . -iname "tags" -exec rm '{}' ';'
sudo rm -r ./tools/pybrain/build/
