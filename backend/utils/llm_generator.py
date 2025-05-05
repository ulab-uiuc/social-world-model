import base64
import json
import os
from typing import Any, Dict, List, Optional, Union

import litellm
from together import Together


class LLMGenerator:
    def __init__(self, model_name: str = 'gpt-4o'):
        self.model_name = model_name
        self.use_together_native = False
        self._setup_api_keys()

        if any(
            key in model_name.lower()
            for key in ['together', 'deepseek', 'qwen', 'llama']
        ):
            self.use_together_native = True
            self.together_model = (
                model_name.replace('together/', '')
                if 'together/' in model_name
                else model_name
            )
            self.together_client = Together()
            self.together_client.api_key = os.getenv('TOGETHER_API_KEY')

    def _setup_api_keys(self):
        if 'gpt' in self.model_name:
            litellm.api_key = os.getenv('OPENAI_API_KEY')
        elif any(k in self.model_name for k in ['together', 'deepseek', 'qwen']):
            litellm.api_key = os.getenv('TOGETHER_API_KEY')

    def build_prompt(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        if not system_prompt:
            system_prompt = (
                'You are a helpful assistant skilled in step-by-step reasoning.'
            )
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    def generate(
        self, prompt: Union[str, List], system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        if isinstance(prompt, str):
            messages = self.build_prompt(prompt, system_prompt)
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError('Prompt must be a string or a list of chat messages.')

        try:
            if self.use_together_native:
                response = self.together_client.chat.completions.create(
                    model=self.together_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
                return {
                    'content': response.choices[0].message.content,
                    'usage': getattr(response, 'usage', {}),
                }
            else:
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
                return {
                    'content': response.choices[0].message['content'].strip(),
                    'usage': response.get('usage', {}),
                }

        except Exception as e:
            print(f'LLM generation error: {e}')
            return {'content': json.dumps({'error': str(e), 'answer': ''}), 'usage': {}}

    def generate_batch(
        self, prompts: List[str], system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [self.generate(p, system_prompt) for p in prompts]

    def load_image(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f'Failed to load image: {e}')
            return None

    def generate_with_image(
        self, text: str, image_paths: List[str], system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        message_content = [{'type': 'text', 'text': text}]
        for path in image_paths:
            encoded = self.load_image(path)
            if encoded:
                message_content.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{encoded}'},
                    }
                )

        if self.use_together_native:
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt or 'You are a helpful assistant.',
                },
                {'role': 'user', 'content': message_content},
            ]
            try:
                response = self.together_client.chat.completions.create(
                    model=self.together_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
                return {
                    'content': response.choices[0].message.content,
                    'usage': getattr(response, 'usage', {}),
                }
            except Exception as e:
                print(f'Multimodal generation error: {e}')
                return {'content': '', 'usage': {}}
        else:
            print('Image input is only supported with Together-native models.')
            return {'content': '', 'usage': {}}
