from collections import defaultdict
from typing import List, Set, Union

from .utils import timed


class WordTrieNode:
    """
    @author Phil McLaughlin (https://github.com/pmclaugh)
    """
    def __init__(self, value: str, parent: Union["WordTrieNode", None]):
        self.value = value
        self.parent = parent
        self.children = {}
        self.valid = False

    def get_word(self) -> str:
        if self.parent is not None:
            return self.parent.get_word() + self.value
        else:
            return self.value


class LetterBoxed:
    """
    @author Phil McLaughlin (https://github.com/pmclaugh)
    """
    @timed
    def __init__(self, input_string: str, dictionary: str, len_threshold=3):
        # parse the input string (abc-def-ghi-jkl) into set of 4 sides
        self.input_string = input_string.lower()
        self.sides = {side for side in input_string.split("-")}
        self.puzzle_letters = {letter for side in self.sides for letter in side}
        self.len_threshold = len_threshold

        # build trie from newline-delimited .txt word list
        self.root = WordTrieNode("", None)
        with open(dictionary) as f:
            for line in f.readlines():
                self.add_word(line.strip().lower())

        # find all valid words in puzzle
        self.puzzle_words = self.get_puzzle_words()

        # puzzle_graph[starting_letter][ending_letter] = {{letters}: [words]}
        # e.g. puzzle_graph['f']['s'] = {{'a','e','f','r','s'} : ['fares', 'fears', 'farers', 'fearers']}
        self.puzzle_graph = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for word in self.puzzle_words:
            self.puzzle_graph[word[0]][word[-1]][frozenset(word)].append(word)

    def add_word(self, word) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = WordTrieNode(char, node)
            node = node.children[char]
        node.valid = True

    def _puzzle_words_inner(
        self, node: WordTrieNode, last_side: str
    ) -> List[WordTrieNode]:
        valid_nodes = [node] if node.valid else []
        if node.children:
            for next_side in self.sides - {last_side}:
                for next_letter in next_side:
                    if next_letter in node.children:
                        next_node = node.children[next_letter]
                        valid_nodes += self._puzzle_words_inner(next_node, next_side)
        return valid_nodes

    @timed
    def get_puzzle_words(self) -> List[str]:
        all_valid_nodes = []
        for starting_side in self.sides:
            for starting_letter in starting_side:
                if starting_letter in self.root.children:
                    all_valid_nodes += self._puzzle_words_inner(
                        self.root.children[starting_letter], starting_side
                    )
        return [node.get_word() for node in all_valid_nodes]

    def _find_solutions_inner(
        self, path_words: List[List[str]], letters: Set[str], next_letter: str
    ) -> List[List[List[str]]]:
        if len(letters) == 12:
            return [path_words]
        elif len(path_words) == self.len_threshold:
            return []

        solutions = []
        for last_letter in self.puzzle_graph[next_letter]:
            for letter_edge, edge_words in self.puzzle_graph[next_letter][
                last_letter
            ].items():
                if letter_edge - letters:
                    solutions += self._find_solutions_inner(
                        path_words + [edge_words], letters | letter_edge, last_letter
                    )
        return solutions

    @timed
    def find_all_solutions(self) -> List[List[str]]:
        all_solutions = []
        for first_letter in self.puzzle_letters:
            for last_letter in self.puzzle_letters:
                for letter_edge, edge_words in self.puzzle_graph[first_letter][
                    last_letter
                ].items():
                    all_solutions += self._find_solutions_inner(
                        [edge_words], letter_edge, last_letter
                    )
        return all_solutions


def puzzle_from_string(input_string: str) -> str:
    """
    Parse puzzle from string of format
    CMU
    ZOH
    SBI
    RAN
    """
    input_string = input_string.replace("\n", "-").lower().strip()
    
    if len(input_string) - 3 < 12:
        raise ValueError("not enough letters for correct puzzle")

    return input_string


def solve(puzzle: str, accepted_len: tuple[int, int] = (3, 6)) -> tuple[int, int, List[List[str]]]:
    puzzle = puzzle_from_string(puzzle)
    puzzle = LetterBoxed(puzzle, "solver/words.txt", accepted_len[0])
    
    meta_solutions = []
    len_threshold = accepted_len[0] - 1
    while not len(meta_solutions) and len_threshold < accepted_len[1]:
        len_threshold += 1
        puzzle.len_threshold = len_threshold
        meta_solutions = puzzle.find_all_solutions()

    full_count = 0
    for meta_solution in meta_solutions:
        count = 1
        for element in meta_solution:
            count *= len(element)
        full_count += count

    return len_threshold, full_count, meta_solutions
