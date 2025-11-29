import os
import requests
import logging
import json
import google.generativeai as genai

def get_gemini_response(prompt, text):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logging.error("Google API key not found")
        return "", {}
    try:
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt, text])
        usage = {}
        try:
            # Attempt to access usage metadata if available
            if hasattr(response, 'usage_metadata'):
                usage = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
        except Exception:
            pass
        return response.text, usage
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "", {}

def list_ollama_models():
    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        r = requests.get(f"{ollama_base_url}/api/tags")
        r.raise_for_status()
        data = r.json()
        return [m.get("model", "") for m in data.get("models", [])]
    except Exception:
        return []

def get_ollama_response(prompt, text, model):
    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {"model": model, "prompt": f"{prompt}\n\n{text}", "stream": False}
        r = requests.post(f"{ollama_base_url}/api/generate", json=payload)
        if r.status_code >= 400:
            try:
                err = r.json().get("error")
            except Exception:
                err = r.text
            logging.error(f"Ollama error: {err}")
            return "", {}
        
        resp_json = r.json()
        usage = {
            'prompt_tokens': resp_json.get('prompt_eval_count', 0),
            'completion_tokens': resp_json.get('eval_count', 0),
            'total_tokens': resp_json.get('prompt_eval_count', 0) + resp_json.get('eval_count', 0)
        }
        return resp_json.get("response", ""), usage
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama connect error: {e}")
        return "", {}
    except Exception as e:
        logging.error(f"Ollama generate error: {e}")
        return "", {}

def get_gemini_stream(prompt, text, usage_callback=None):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        yield ""
        return
    try:
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt, text], stream=True)
        for chunk in response:
            yield chunk.text
        
        if usage_callback and hasattr(response, 'usage_metadata'):
             usage = {
                'prompt_tokens': response.usage_metadata.prompt_token_count,
                'completion_tokens': response.usage_metadata.candidates_token_count,
                'total_tokens': response.usage_metadata.total_token_count
            }
             usage_callback(usage)
    except Exception:
        yield ""

def get_ollama_stream(prompt, text, model, usage_callback=None):
    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {"model": model, "prompt": f"{prompt}\n\n{text}", "stream": True}
        with requests.post(f"{ollama_base_url}/api/generate", json=payload, stream=True) as r:
            if r.status_code >= 400:
                try:
                    err = r.json().get("error")
                except Exception:
                    err = r.text
                logging.error(f"Ollama error: {err}")
                yield ""
                return
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode('utf-8'))
                except Exception:
                    continue
                s = obj.get('response')
                if s:
                    yield s
                if obj.get('done'):
                    if usage_callback:
                        usage = {
                            'prompt_tokens': obj.get('prompt_eval_count', 0),
                            'completion_tokens': obj.get('eval_count', 0),
                            'total_tokens': obj.get('prompt_eval_count', 0) + obj.get('eval_count', 0)
                        }
                        usage_callback(usage)
                    break
    except Exception as e:
        logging.error(f"Ollama stream error: {e}")
        yield ""