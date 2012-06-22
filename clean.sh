#!/usr/bin/env sh
rm ./res/*
find . -iname "*.pyc" -exec rm '{}' ';'
find . -iname "tags" -exec rm '{}' ';'
