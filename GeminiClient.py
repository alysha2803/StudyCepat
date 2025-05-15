import google.generativeai as genai
import re
import json

class GeminiClient:
    def __init__(self, api_key, model_name="gemini-2.0-flash-001"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_content(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating content: {e}")

    def generate_flashcards(self, summary):
        try:
            # Try to get the model to return JSON directly
            prompt = f"""
            You are a flashcard generator. Create flashcards from the following summary. Based on the length and complexity of the content,
            generate an appropriate number of flashcards (5-15) that cover the key concepts.

            Summary:
            {summary}

            Return ONLY a JSON array of objects with 'question' and 'answer' fields. The JSON array MUST be valid and parsable.
            Format example:
            [
                {{"question": "What is X?", "answer": "X is Y"}},
                {{"question": "Who discovered Z?", "answer": "Z was discovered by W"}}
            ]

            Do not include any explanatory text, markdown formatting, or anything else outside of the JSON array.
            Ensure that the question and answer are concise and clear.
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to parse as JSON first
            try:
                # Find JSON-like content if it's embedded in other text
                json_pattern = r'\[.*\{.*"question".*"answer".*\}.*\]'
                json_match = re.search(json_pattern, response_text, re.DOTALL)

                if json_match:
                    response_text = json_match.group(0)

                # Clean up common JSON formatting issues
                response_text = response_text.replace("'", '"')  # Replace single quotes with double quotes

                flashcards = json.loads(response_text)
                if isinstance(flashcards, list) and len(flashcards) > 0:
                    # Validate the structure
                    valid_flashcards = []
                    for card in flashcards:
                        if isinstance(card, dict) and "question" in card and "answer" in card:
                            valid_flashcards.append({
                                "question": card["question"].strip(),
                                "answer": card["answer"].strip()
                            })
                    
                    if valid_flashcards:
                        return valid_flashcards
            except json.JSONDecodeError:
                print("JSON parsing failed, falling back to simplified regex pattern matching")

            # Fallback to simplified regex pattern matching
            flashcards = []

            # Try the simplified pattern
            simplified_pattern = r'"question":\s*"(.*?)"\s*,\s*"answer":\s*"(.*?)"'
            matches = re.findall(simplified_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    question = match[0].strip()
                    answer = match[1].strip()
                    if question and answer:  # Ensure neither is empty
                        flashcards.append({"question": question, "answer": answer})

            # If we still have no flashcards, create a fallback
            if not flashcards:
                # Create a single fallback flashcard with some content from the summary
                summary_snippet = summary[:100] + "..." if len(summary) > 100 else summary
                flashcards = [
                    {"question": "What is the main topic of this document?",
                     "answer": f"The document discusses: {summary_snippet}"},
                    {"question": "What might you want to learn from this document?",
                     "answer": "To understand the key concepts and information contained within."}
                ]

            return flashcards
        
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            # Return at least some flashcards even in error case
            return [
                {"question": "What might be the main topic of this document?", 
                 "answer": "The document likely discusses important concepts related to the subject matter."},
                {"question": "Why are flashcards useful for learning?",
                 "answer": "Flashcards help with active recall, which strengthens memory and improves learning retention."}
            ]