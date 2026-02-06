"""
Direct Competition Environment

This environment controls the evaluation of all agents.
Implement the `evaluate` method to run your competition logic.
"""

class Env:
    def __init__(self):
        """
        Initialize the environment.
        Load any data or resources needed for evaluation here.
        """
        pass

    def evaluate(self, agents: list, agent_infos: list) -> dict:
        """
        Evaluate all agents and return their scores.

        Args:
            agents: List of instantiated Agent classes from participants
            agent_infos: List of dicts with agent metadata:
                - agent_index: int - Position in agents list
                - agent_id: int - Database ID
                - agent_name: str - Name given by participant
                - user_name: str - Username of participant
                - attach_path_files: list - Additional files uploaded
                - model_filename: str - Optional model file name

        Returns:
            dict with structure:
            {
                "agent_results": [
                    {
                        "agent_index": 0,      # Required: matches agent position
                        "score": 0.95,         # Required: main score
                        "score2": 0.0,         # Optional: secondary metric
                        "steps": 1,            # Optional: number of steps
                        "is_agent_code_error": False,
                        "agent_code_error_message": None,
                        "is_agent_simulation_error": False,
                        "agent_simulation_error_message": None,
                        "info_message": None   # Optional: display message
                    },
                    # ... one entry per agent
                ]
            }

        Example:
            results = []
            for i, (agent, info) in enumerate(zip(agents, agent_infos)):
                try:
                    # Call agent's method (define what you expect from agents)
                    prediction = agent.predict(self.test_data)
                    score = self.calculate_score(prediction)
                    results.append({
                        "agent_index": i,
                        "score": score
                    })
                except Exception as e:
                    results.append({
                        "agent_index": i,
                        "score": 0.0,
                        "is_agent_code_error": True,
                        "agent_code_error_message": str(e)
                    })
            return {"agent_results": results}
        """
        # TODO: Implement your evaluation logic
        results = []
        for i, (agent, info) in enumerate(zip(agents, agent_infos)):
            results.append({
                "agent_index": i,
                "score": 0.0,
                "info_message": "Evaluation not implemented yet"
            })
        return {"agent_results": results}
