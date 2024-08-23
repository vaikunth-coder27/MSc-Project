# Copyright (c) Microsoft Corporation.
# Copyright (c) 2023 Konstantin Chernyshev.
# Licensed under the MIT license.
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from . import bleu, dataflow_match, syntax_match, weighted_ngram_match
from .utils import AVAILABLE_LANGS, get_tree_sitter_language

PACKAGE_DIR = Path(__file__).parent


def calc_codebleu(
    references: Union[List[str], List[List[str]]],
    predictions: List[str],
    lang: str,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    tokenizer: Optional[Callable] = None,
    keywords_dir: Path = PACKAGE_DIR / "keywords",
) -> Dict[str, float]:
    """Calculate CodeBLEU score

    Args:
        predictions: list of predictions
        references: list of lists with references
        lang: input language, one of AVAILABLE_LANGS
        weights: weights of the ngram_match, weighted_ngram_match, syntax_match, and dataflow_match respectively
        tokenizer: tokenizer function, Defaults to lambda s: s.split()
        keywords_dir: path to the directory with keywords files
        lang_so_file: path to the .so file with the parser for the language

    Return:
        Scores dict
    """
    assert len(references) == len(predictions), "Number of references and predictions should be the same"
    assert lang in AVAILABLE_LANGS, f"Language {lang} is not supported (yet). Available languages: {AVAILABLE_LANGS}"
    assert len(weights) == 4, "weights should be a tuple of 4 floats (alpha, beta, gamma, theta)"
    assert keywords_dir.exists(), f"keywords_dir {keywords_dir} does not exist"

    # get the tree-sitter language for a given language
    tree_sitter_language = get_tree_sitter_language(lang)

    # preprocess inputs
    references = [[x.strip() for x in ref] if isinstance(ref, list) else [ref.strip()] for ref in references]
    hypothesis = [x.strip() for x in predictions]

    # calculate ngram match (BLEU)
    if tokenizer is None:

        def tokenizer(s):
            return s.split()

    tokenized_hyps = [tokenizer(x) for x in hypothesis]
    tokenized_refs = [[tokenizer(x) for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    with open(keywords_dir / (lang + ".txt"), "r", encoding="utf-8") as f:
        keywords = [x.strip() for x in f.readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [
        [[reference_tokens, make_weights(reference_tokens, keywords)] for reference_tokens in reference]
        for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(
        references, hypothesis, lang, tree_sitter_language=tree_sitter_language
    )

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, lang, tree_sitter_language=tree_sitter_language
    )

    alpha, beta, gamma, theta = weights
    code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * (dataflow_match_score or 1)
    )

    return {
        "codebleu": code_bleu_score,
        "ngram_match_score": ngram_match_score,
        "weighted_ngram_match_score": weighted_ngram_match_score,
        "syntax_match_score": syntax_match_score,
        "dataflow_match_score": dataflow_match_score,
    }

from analyze import *

analyze_code_function = {'python': analyze_python_code, 'java': analyze_java_code, 'javascript': analyze_javascript_code, 'ruby': analyze_ruby_code}

def compare_metrics(references, predictions,lang):

    references_dict = analyze_code_function[lang](references)
    predictions_dict = analyze_code_function[lang](predictions)
    score = 0
    total_count = 0
    # print(references_dict)
    # print(predictions_dict)
    weights = {'variable_names': 0.5, 'variable_values': 0.2, 'strings': 0.1, 'comments': 0.2}
    for key1,value1,key2,value2 in zip(references_dict.keys(),references_dict.values(),predictions_dict.keys(),predictions_dict.values()):
        if key1 == key2:
            if len(value1) > len(value2):
                l=value1
                c = value2
            else:
                l=value2
                c = value1
            for i in l:
                if i in c:
                    score += 1#weights[key1]
                total_count += 1
        
    # print(score,total_count)
    return score/total_count



def modified_calc_codebleu(
    references: Union[List[str], List[List[str]]],
    predictions: List[str],
    lang: str,
    weights: Tuple[float, float, float, float,float] = (0.15, 0, 0.35, 0.15, 0.35),
    tokenizer: Optional[Callable] = None,
    keywords_dir: Path = PACKAGE_DIR / "keywords",
) -> Dict[str, float]:
    """Calculate CodeBLEU score

    Args:
        predictions: list of predictions
        references: list of lists with references
        lang: input language, one of AVAILABLE_LANGS
        weights: weights of the ngram_match, weighted_ngram_match, syntax_match, and dataflow_match respectively
        tokenizer: tokenizer function, Defaults to lambda s: s.split()
        keywords_dir: path to the directory with keywords files
        lang_so_file: path to the .so file with the parser for the language

    Return:
        Scores dict
    """
    assert len(references) == len(predictions), "Number of references and predictions should be the same"
    assert lang in AVAILABLE_LANGS, f"Language {lang} is not supported (yet). Available languages: {AVAILABLE_LANGS}"
    assert len(weights) == 5, "weights should be a tuple of 4 floats (alpha, beta, gamma, theta)"
    assert keywords_dir.exists(), f"keywords_dir {keywords_dir} does not exist"

    
    # get the tree-sitter language for a given language
    tree_sitter_language = get_tree_sitter_language(lang)

    variable_match_score = compare_metrics(references[0], predictions[0],lang)
    # preprocess inputs
    references = [[x.strip() for x in ref] if isinstance(ref, list) else [ref.strip()] for ref in references]
    hypothesis = [x.strip() for x in predictions]

    # calculate ngram match (BLEU)
    if tokenizer is None:

        def tokenizer(s):
            return s.split()

    tokenized_hyps = [tokenizer(x) for x in hypothesis]
    tokenized_refs = [[tokenizer(x) for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    with open(keywords_dir / (lang + ".txt"), "r", encoding="utf-8") as f:
        keywords = [x.strip() for x in f.readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [
        [[reference_tokens, make_weights(reference_tokens, keywords)] for reference_tokens in reference]
        for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(
        references, hypothesis, lang, tree_sitter_language=tree_sitter_language
    )

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, lang, tree_sitter_language=tree_sitter_language
    )



    alpha, beta, gamma, theta,swag = weights

    code_bleu_score = (
        0.25 * ngram_match_score
        + 0.25 * weighted_ngram_match_score
        + 0.25 * syntax_match_score
        + 0.25 * (dataflow_match_score or 1)
    )

    new_code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * (dataflow_match_score or 1)
        + swag * variable_match_score
    )

    return {
        "codebleu": code_bleu_score,
        "modified_codebleu": new_code_bleu_score,
        "ngram_match_score": ngram_match_score,
        "weighted_ngram_match_score": weighted_ngram_match_score,
        "syntax_match_score": syntax_match_score,
        "dataflow_match_score": dataflow_match_score,
        "variable_match_score": variable_match_score
    }
