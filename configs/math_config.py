import re
from functools import lru_cache

class MathConfig:
    # Regex pattern for extracting numbers
    NUMBER_PATTERN = re.compile(r"\-?\d+\.\d+|\-?\d+")

    # Answer formats for different types of problems
    ANSWER_FORMATS = {
        "gsm": r'"[Final Answer] The answer is [answer] \n#### [answer]"',
        "number": r'"[Final Answer] The answer is [number] \n#### [number]"',
        "option": r'"[Final Answer] The answer is \boxed{[option]} \n#### [option]"',
        "yesno": r'"[Final Answer] The answer is \boxed{[Yes or No]} \n#### [Yes or No]"',
        "formula": r'"[Final Answer] The answer is \boxed{[answer formula]} \n#### [answer formula]"'
    }

    # Base prompts
    PROMPTS = {
        "weak_answer": "Question: {question}\nThe response should begin with [reasoning process]...[Verification]... and end with {ans_format}\nLet's think step by step.",
        
        "weak_hints": "Question: {question}\nAnalyze the answer to the provided question rigorously and critically. Identify every logical flaw or misstep in the reasoning process that contributed to the answer being suboptimal. Highlight areas where the reasoning can be improved, ensuring each issue is clearly explained. Provide actionable hints and suggestions to refine and improve the answer. Address all aspects of the reasoning process step-by-step.",
        
        "better_answer": "Question: {question}\nPlease refine your answer according to the feedback provided. The response should begin with [reasoning process]...[Verification]... and end with end with {ans_format}\nLet's think step by step.",
        
        "reward_calculation": """
        Question: {question}\nAnswer:{answer}\n Strictly critic and analyze the answer to the provided question. Point out every logical flaw that was made in the reasoning process that resulted in the final answer being incorrect under [Analysis]. Output a score between -100 to +100 that represents the quality of this answer based on the logical errors made in the reasoning process under [Score].
        Use the following rubric to assign scores:
        +75 to +100:
        No logical errors were made, the answer correctly interpreted the question and took the correct approach to solving the problem.
        +0 to +74:
        Minor logical errors made. Mistakes that have been made can be corrected during the revision process.
        -74 to -1:
        Major logical errors made. Mistakes that have been made can be corrected during the revision process.
        -100 to -75:
        Many major logical errors were made. Incorrect approach taken or misinterpreted question. 
        Use the following format for your response\nResponse format:\n[Analyst]...[Score]..."""
    }

    @staticmethod
    def fix_fracs(string):
        """Fix fraction notation in LaTeX strings"""
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        return new_str

    @staticmethod
    def fix_sqrt(string):
        """Fix square root notation in LaTeX strings"""
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    @staticmethod
    def strip_string(string):
        """Clean and standardize math expressions"""
        # Skip empty strings
        if not string:
            return string

        # Basic cleanup
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        string = string.replace("\\%", "")
        string = string.replace("\%", "")
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        
        # Handle leading decimal
        if string and string[0] == ".":
            string = "0" + string

        # Handle equals signs
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # Fix notation
        string = MathConfig.fix_sqrt(string)
        string = string.replace(" ", "")
        string = MathConfig.fix_fracs(string)
        
        # Special cases
        if string == "0.5":
            string = "\\frac{1}{2}"
            
        string = string.replace("x \\in", "").strip()

        # Handle subscripts
        if string.find("_") >= 0:
            p = string.split("_")
            p[1] = p[1].replace("{", "").replace("}", "")
            string = "_".join(p)

        # Remove commas from numbers
        if string.strip().find(" ") == -1 and string.find("(") == -1:
            string = string.replace(",", "")

        return string

    @staticmethod
    @lru_cache(1024)
    def extract_answer(text: str, answer_type="") -> str:
        """Extract answer from text based on type"""
        if "gsm" not in answer_type and answer_type != "digit":
            if "####" in text:
                text = text.split("####")[-1]
            elif "The answer is" in text:
                text = text.split("The answer is")[-1]
                if "####" in text:
                    text = text.split("####")[-1]
            if "box" in text:
                return MathConfig.extract_boxed_answer(text)
            else:
                return text

        if "\n####" in text:
            text = text.split("\n####")[-1].replace(",", "")
        elif "The answer is" in text:
            text = text.split("The answer is")[-1].replace(",", "")
            
        numbers = MathConfig.NUMBER_PATTERN.findall(text)
        if not numbers:
            return None
            
        if "\n####" in text or "The answer is" in text:
            return numbers[0]
        else:
            return numbers[-1]

    @staticmethod
    def extract_boxed_answer(text):
        """Extract answer from LaTeX boxed notation"""
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        
        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            if text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None
            
        boxed = text[idx:right_brace_idx + 1]
        try:
            left = "\\boxed{"
            assert boxed[:len(left)] == left
            assert boxed[-1] == "}"
            return boxed[len(left):-1]
        except:
            return None

    @staticmethod
    def is_equivalent(str1, str2, verbose=False):
        """Check if two math expressions are equivalent"""
        if str1 is None and str2 is None:
            return False
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = MathConfig.strip_string(str1)
            ss2 = MathConfig.strip_string(str2)
            return ss1 == ss2
        except:
            return str1 == str2

    @staticmethod
    def check_answer(ground_truth, answer, dataset_type=""):
        """Check if an answer matches the ground truth
        
        Args:
            ground_truth: The correct answer string
            answer: The provided answer string
            dataset_type: The type of dataset (e.g. "gsm")
        
        Returns:
            bool: Whether the answer is correct
        """
        gt_label = MathConfig.extract_answer(ground_truth)
        if not gt_label:
            return False
            
        # Determine answer type
        if gt_label.isdigit():
            answer_type = "digit"
        elif gt_label.isupper() and gt_label.isalpha():
            answer_type = "option"
        elif gt_label.lower() in ["yes", "no"]:
            answer_type = "yesno"
        else:
            answer_type = "formula"
            
        # Extract and format answer
        ans_label = MathConfig.extract_answer(answer, answer_type)
        if not ans_label:
            return False
            
        # Format based on type
        if answer_type == "option":
            ans_label = ans_label.strip()[0]
        elif answer_type == "yesno":
            ans_label = ans_label.lower()
            gt_label = gt_label.lower()
        elif answer_type == "formula":
            ans_label = ans_label.replace("$", "")
            
        # Check for match
        if "gsm" not in dataset_type and answer_type != "digit":
            return MathConfig.is_equivalent(gt_label, ans_label)
            
        try:
            return (ans_label == gt_label or 
                   abs(float(ans_label) - float(gt_label)) < 1e-5)
        except (ValueError, TypeError):
            return False
