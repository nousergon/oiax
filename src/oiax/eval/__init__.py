"""Evaluation harness — measures miss rate against labelled ground truth.

The live router counts firings. The eval measures what should have fired
and didn't — the false negatives the delivery layer itself cannot see.

Judge labels are evidence, not proof. The acceptance bar is 0 false
positives on the negative set (oiax positioning doc §4 decision 2).
"""
