import os
import sys
import hashlib
import json
import time
import random
from functools import lru_cache
import numpy as np
from tqdm import tqdm
from retry import retry
from datasets import load_dataset
from litellm import completion

class Question:
    def __init__(self, question, ground_truth, config, data_name):
        self.config = config
        self.data_name = data_name

        self.question = question
        self.answer = self.extract_answer(ground_truth)
        self.answer_explanation = ground_truth
        self.answer_format = self.get_answer_format(self.answer)

    def extract_answer(self, answer):
        """Extract answer using config's extraction method"""
        return self.config.extract_answer(answer)

    def check_answer(self, answer):
        """Check if answer matches ground truth using config's check method"""
        return self.config.check_answer(self.answer, answer, self.data_name)

    def get_answer_format(self, gt_label):
        """Determine appropriate answer format"""
        if "gsm8k" in self.data_name:
            return self.config.ANSWER_FORMATS["gsm"]
        elif gt_label.isdigit():
            return self.config.ANSWER_FORMATS["number"]
        elif gt_label.isalpha() and gt_label.isupper():
            return self.config.ANSWER_FORMATS["option"]
        elif gt_label.lower() in ["yes", "no"]:
            return self.config.ANSWER_FORMATS["yesno"]
        else:
            return self.config.ANSWER_FORMATS["formula"]

class Runner:
    def __init__(self, config_class, model_name="", data_name=""):
        """Initialize runner with specific config"""
        self.config = config_class
        self.model_name = model_name or sys.argv[1]
        self.data_name = data_name or sys.argv[2]
        self.clients = []
        self.pattern = self.config.NUMBER_PATTERN
        self.dataset = self.setup_dataset()
        self.setup_environment()

    def setup_environment(self):
        """Initialize API keys and create necessary directories"""
        api_keys = [
            os.getenv("CUSTOM_LLM_API_KEY"),
        ]
        self.clients = [key for key in api_keys if key]
        if not self.clients:
            raise Exception("No API keys found")

        if not os.path.exists(self.data_name):
            os.mkdir(self.data_name)
        if not os.path.exists(f"{self.data_name}/jsons"):
            os.mkdir(f"{self.data_name}/jsons")

    def setup_dataset(self):
        """Load and prepare dataset based on data_name"""
        print(self.data_name)
        if "gsm8k" in self.data_name:
            raw_dataset = load_dataset("gsm8k", "main", split="test")
            raw_dataset = list(zip(raw_dataset["question"], raw_dataset["answer"]))
        elif 'AIME' in self.data_name:
            raw_dataset = load_dataset("qq8933/AIME_1983_2024", split='train')
            raw_dataset = list(zip(raw_dataset["Question"], raw_dataset["Answer"]))
        elif 'GPQA' in self.data_name:
            raw_dataset = load_dataset("qq8933/GPQA", split='train')
            raw_dataset = list(zip(raw_dataset["question"], raw_dataset["answer"]))
        else:
            raise ValueError(f"No valid dataset found for {self.data_name}")
        
        raw_dataset = raw_dataset[:5]
        print(len(raw_dataset))
        
        processed_dataset = []
        for q, a in raw_dataset:
            question = Question(q, a, self.config, self.data_name)
            processed_dataset.append(question)
        
        return processed_dataset

    @retry()
    def generate(self, prompt, history=[], timeout=300, truncate=True):
        """Generate response using LLM"""
        if "testtime" in self.data_name:
            timeout = 150
        print("awaiting response...")
        time0 = time.time()

        history_ = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": h}
            for i, h in enumerate(history)
        ]
        if truncate:
            history_ = history_[-2:]

        try:
            response = completion(
                model=self.model_name,
                messages=history_ + [{"role": "user", "content": prompt}],
                temperature=0.95,
                timeout=timeout,
                base_url="http://avior.mlfoundry.com/live-inference/v1",
                api_key=random.choice(self.clients),
            )
            print(f"response received! time taken: {time.time()-time0} seconds.")
            return response.choices[0].message.content, list(history) + [
                prompt,
                response.choices[0].message.content,
            ]
        except Exception as e:
            print(f"Error generating response: {e}")
            raise

    @retry()
    def cal_reward(self, question, ans):
        """Calculate reward for an answer"""
        query = self.config.PROMPTS["reward_calculation"].format(
            question=question,
            answer=ans
        )
        ret = self.generate(query)
        score = ret[0].split("Score")[-1]
        scores = self.pattern.findall(score)
        if not scores:
            raise Exception("no score found")
        ret_score = float(scores[-1])
        return ret_score, ret[0]

    def get_weak_answer(self, question, ans_format=""):
        """Get initial answer attempt"""
        query = self.config.PROMPTS["weak_answer"].format(
            question=question,
            ans_format=ans_format
        )
        return self.generate(query, timeout=300)

    def get_weak_hints(self, question, weak_answer, history=[]):
        """Get hints for improving the answer"""
        query = self.config.PROMPTS["weak_hints"].format(question=question)
        return self.generate(query, history)

    def get_better_answer(self, question, weak_answer, hint, history=[], ans_format=""):
        """Get improved answer based on hints"""
        query = self.config.PROMPTS["better_answer"].format(
            question=question,
            ans_format=ans_format
        )
        return self.generate(query, history)

    def process_question(self, question: Question):
        """Process a single example"""
        file_path = f"{self.data_name}/jsons/{hashlib.md5(str(question.question).encode()).hexdigest()}.json"
        if os.path.exists(file_path):
            return {}
        
        query = question.question
        ground_truth = question.answer
        answer_format = question.answer_format
        max_iter = 2

        results = self.main_loop(query, ground_truth, answer_format, max_iter)

        with open(file_path, "w+") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    def main_loop(self, query, ground_truth, ans_format, max_iter):
        """Main iteration loop for processing a question"""
        ret = {
            "reward_analysis": {},
            "ucb_bank": {},
            "to_explore": [],
            "to_explore_reward": {},
            "history_bank": {},
            "hints_reward_imp_bank": {},
            "hints_bank": {},
            "hints_list": [],
            "fathers": {},
            "childs": {},
            "correct_answers": [],
            "answers_list": [],
        }

        def sampling_reward(answer):
            if answer not in ret["to_explore_reward"]:
                ret["to_explore_reward"][answer] = []
            if answer not in ret["reward_analysis"]:
                ret["reward_analysis"][answer] = []
            score, analysis = self.cal_reward(query, answer)
            ret["to_explore_reward"][answer].append(score)
            ret["reward_analysis"][answer].append(analysis)

        def add_to_hints_bank(hints, weak_answer):
            if weak_answer not in ret["hints_bank"]:
                ret["hints_bank"][weak_answer] = []
            ret["hints_bank"][weak_answer].append(hints)

        def add_to_childs(father, child):
            if father not in ret["childs"]:
                ret["childs"][father] = []
            ret["childs"][father].append(child)

        def add_to_hints_reward_imp_bank(hints, weak_answer, reward, answer):
            if weak_answer not in ret["hints_reward_imp_bank"]:
                ret["hints_reward_imp_bank"][weak_answer] = []
            ret["hints_reward_imp_bank"][weak_answer].append((hints, reward, answer))

        def compute_ucb(r_c, N_n, N_c, C=1.4):
            return r_c + C * np.sqrt(np.log(N_n + 1) / (N_c + 1e-5))

        def update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4):
            visit_count = {node: len(to_explore_reward[node]) for node in to_explore}
            avg_reward = {
                node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2
                for node in to_explore
            }
            
            leaves = set(to_explore) - set(fathers.values())
            
            for leaf in leaves:
                ucb_bank[leaf] = compute_ucb(
                    avg_reward[leaf],
                    len(to_explore_reward.get(fathers.get(leaf, None), [])),
                    len(to_explore_reward.get(leaf, [])),
                    C,
                )

            nodes_to_update = list(leaves)
            while nodes_to_update:
                new_nodes_to_update = set()
                for node in nodes_to_update:
                    father = fathers.get(node)
                    if father is not None:
                        if father not in ucb_bank:
                            new_nodes_to_update.add(father)
                        if father in ucb_bank:
                            child_reward = []
                            for child in childs[father]:
                                child_reward.append(avg_reward[child])
                            father_reward = (avg_reward[father] + max(child_reward)) / 2
                            ucb_bank[father] = compute_ucb(
                                father_reward,
                                len(to_explore_reward.get(fathers.get(father, None), [])),
                                len(to_explore_reward.get(father, [])),
                                C,
                            )
                nodes_to_update = list(new_nodes_to_update)

        def filter_mature_node(childs, to_explore, to_explore_reward, max_expand=3):
            filtered_to_explore = []
            avg_reward = {
                node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2
                for node in to_explore
            }

            for node in to_explore:
                if len(childs.get(node, [])) < max_expand or max(
                    [avg_reward.get(child, -999) for child in childs.get(node, [])]
                ) < avg_reward.get(node, -999):
                    filtered_to_explore.append(node)

            return filtered_to_explore

        def get_best_explore_from_ucb(to_explore, ucb_bank):
            best_node = None
            highest_ucb = float("-inf")
            
            for node in to_explore:
                ucb_value = ucb_bank.get(node, float("-inf"))
                if ucb_value > highest_ucb:
                    highest_ucb = ucb_value
                    best_node = node
            
            return best_node

        # Get initial weak answer

        weak_answer, history = self.get_weak_answer(query, ans_format=ans_format)
        if self.config.check_answer(ground_truth, weak_answer, self.data_name):
            ret["correct_answers"].append(weak_answer)
        ret["history_bank"][weak_answer] = tuple(history)
        ret["answers_list"] = [weak_answer]
        ret["to_explore"] = [weak_answer]
        ret["childs"][weak_answer] = []
        ret["fathers"][weak_answer] = None
        
        sampling_reward(weak_answer)

        if self.config.check_answer(ground_truth, weak_answer, self.data_name):
            return ret

        patient = 0 if "testtime" not in self.data_name else 0
        alpha = 0.45

        update_ucb(ret["fathers"], ret["childs"], ret["to_explore"], ret["to_explore_reward"], ret["ucb_bank"])

        for i in range(max_iter):
            print("iteration:", i)

            # Select node to explore
            filtered_to_explore = filter_mature_node(ret["childs"], ret["to_explore"], ret["to_explore_reward"])
            weak_answer = get_best_explore_from_ucb(filtered_to_explore, ret["ucb_bank"])
            sampling_reward(weak_answer)

            # Generate hints
            hints, history = self.get_weak_hints(
                query, weak_answer, history=ret["history_bank"][weak_answer]
            )
            # Generate better answer
            answer, history = self.get_better_answer(
                query, weak_answer, hints, history=history, ans_format=ans_format
            )
            
            # Update tracking
            if self.config.check_answer(ground_truth, answer, self.data_name):
                ret["correct_answers"].append(answer)
            add_to_hints_bank(hints, weak_answer)
            ret["history_bank"][answer] = tuple(history)
            ret["to_explore"].append(answer)
            sampling_reward(answer)
            ret["fathers"][answer] = weak_answer
            ret["childs"][answer] = []
            add_to_childs(weak_answer, answer)
            ret["answers_list"].append(answer)
            ret["hints_list"].append(hints)

            if self.config.check_answer(ground_truth, answer, self.data_name):
                if "testtime" in self.data_name:
                    break
                elif patient <= 0:
                    break
                patient -= 1

            update_ucb(ret["fathers"], ret["childs"], ret["to_explore"], ret["to_explore_reward"], ret["ucb_bank"])
            add_to_hints_reward_imp_bank(
                hints,
                weak_answer,
                min(ret["to_explore_reward"].get(answer)) - min(ret["to_explore_reward"].get(weak_answer)),
                answer,
            )

        return ret

    def run(self):
        """Run the processing on the entire dataset"""
        from multiprocessing import Pool
        with Pool(32) as pool:
            pool.map(self.process_question, self.dataset)


if __name__ == "__main__":
    from configs.math_config import MathConfig
    
    runner = Runner(MathConfig)
    runner.run()
