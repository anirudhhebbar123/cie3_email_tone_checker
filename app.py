from flask import Flask, render_template, request, session, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Make transformers optional to avoid startup issues on Render
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
import re
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import Gemini (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


# Initialize Flask app with explicit template and static folders
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production-for-local-use')

# Initialize analyzers with enhanced VADER settings
vader_analyzer = SentimentIntensityAnalyzer()
# Enhance VADER lexicon with email-specific terms
vader_analyzer.lexicon.update({
    'urgent': -0.5, 'asap': -1.5, 'immediately': -1.0,
    'please': 0.5, 'thank you': 1.5, 'thanks': 1.0, 'appreciate': 1.2,
    'regards': 0.3, 'best regards': 0.8, 'sincerely': 0.4,
    'unfortunately': -0.7, 'disappointed': -1.2, 'concerned': -0.5,
    'excellent': 1.5, 'great': 1.2, 'wonderful': 1.4
})
sentiment_pipeline = None
toxicity_pipeline = None
gen_pipeline = None


def get_gemini_api_key():
    """Get Gemini API key"""
    return os.environ.get('GEMINI_API_KEY', '')

def get_openrouter_api_key():
    """Get OpenRouter API key"""
    return os.environ.get('OPENROUTER_API_KEY', '')


def get_generation_pipeline():
    """Lazily load a text-generation pipeline. Returns None on failure.
    Model can be overridden with GEN_MODEL env var (default: google/flan-t5-small).
    """
    global gen_pipeline
    if not TRANSFORMERS_AVAILABLE or pipeline is None:
        return None
    if gen_pipeline is not None:
        return gen_pipeline
    try:
        model_name = os.environ.get('GEN_MODEL', 'google/flan-t5-small')
        gen_pipeline = pipeline('text2text-generation', model=model_name, tokenizer=model_name)
        return gen_pipeline
    except Exception:
        gen_pipeline = None
        return None

def get_pipelines():
    global sentiment_pipeline, toxicity_pipeline
    # Skip if transformers not available
    if not TRANSFORMERS_AVAILABLE or pipeline is None:
        return None, None
    
    # Skip loading heavy transformer models on Render or if explicitly disabled
    # These models are too large for free tier and cause timeouts
    if os.environ.get('DISABLE_TRANSFORMER_MODELS', '').lower() in ('true', '1', 'yes'):
        return None, None
    
    # Skip loading models in production (when PORT is set) unless explicitly enabled
    # This prevents timeouts on Render and other cloud platforms
    if os.environ.get('PORT') and not os.environ.get('ENABLE_TRANSFORMER_MODELS', '').lower() in ('true', '1', 'yes'):
        return None, None
    
    try:
        if sentiment_pipeline is None:
            sentiment_pipeline = pipeline(
                'text-classification',
                model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest',
                top_k=None
            )
        if toxicity_pipeline is None:
            toxicity_pipeline = pipeline(
                'text-classification',
                model='unitary/toxic-bert',
                tokenizer='unitary/toxic-bert',
                top_k=None
            )
    except Exception as e:
        print(f"Error loading pipelines: {e}")
        # Return None if pipelines fail to load - will use fallback methods
        sentiment_pipeline = None
        toxicity_pipeline = None
    return sentiment_pipeline, toxicity_pipeline

# Tone detection keywords
RUDE_KEYWORDS = ['stupid', 'idiot', 'idot', 'fool', 'crazy', 'horrible', 'terrible', 'hate', 
                 'damn', 'hell', 'what the', 'ridiculous', 'absurd', 'pathetic', 'useless',
                 'moron', 'dumb', 'dummy']
FORMAL_KEYWORDS = ['respectfully', 'sincerely', 'regards', 'appreciate', 'kindly', 
                   'regarding', 'pursuant', 'herein', 'whereas', 'aforementioned']
FRIENDLY_KEYWORDS = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'please', 'great', 
                     'awesome', 'love', 'happy', 'excited', 'wonderful', 'cheers']

# Aggressive/imperative patterns that suggest rudeness
AGGRESSIVE_PATTERNS = [
    r'\basap\b',
    r'\bnow\b.*\b(send|do|give|get)',
    r'\bsend\b.*\basap\b',
    r'^(hey|hi|hello)\s+(idiot|stupid|moron|dumb)',
    r'\bdo\s+it\s+now\b',
    r'\b(just|only)\s+(send|do|give)',
]

def truncate_text_for_model(text, max_chars=300):
    """Truncate text to safe length for transformer models (avoid token limit errors)."""
    if not text:
        return text
    
    text = text.strip()
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    last_newline = truncated.rfind('\n')
    
    last_boundary = max(last_period, last_exclamation, last_question, last_newline)
    
    if last_boundary > max_chars * 0.7:
        truncated = truncated[:last_boundary + 1]
    
    return truncated.strip()

def classify_tone(text):
    """Classify tone using transformer models (sentiment + toxicity) and keyword analysis."""
    text = (text or '').strip()
    if not text:
        return 'formal', 0.5, {'compound': 0.0}

    text_lower = text.lower()
    
    text_for_models = truncate_text_for_model(text, max_chars=400)

    rude_keyword_count = sum(1 for kw in RUDE_KEYWORDS if kw in text_lower)
    formal_keyword_count = sum(1 for kw in FORMAL_KEYWORDS if kw in text_lower)
    friendly_keyword_count = sum(1 for kw in FRIENDLY_KEYWORDS if kw in text_lower)
    
    aggressive_pattern_count = sum(1 for pattern in AGGRESSIVE_PATTERNS if re.search(pattern, text_lower, re.IGNORECASE))
    
    hey_used_rudely = False
    if 'hey' in text_lower:
        hey_pattern = re.search(r'\bhey\b\s+(\w+)', text_lower)
        if hey_pattern:
            next_word = hey_pattern.group(1)
            if next_word in ['idiot', 'idot', 'stupid', 'moron', 'dumb']:
                hey_used_rudely = True
            elif re.search(r'\bhey\b.*\b(send|do|give|get)\b', text_lower) and 'please' not in text_lower:
                hey_used_rudely = True

    # Enhanced VADER analysis with context-aware scoring
    vader_scores = vader_analyzer.polarity_scores(text)
    compound_score = vader_scores.get('compound', 0.0)
    
    # Refine compound score based on context and length
    text_length = len(text.split())
    if text_length < 5:
        # Very short texts might have unreliable scores
        compound_score *= 0.7
    elif text_length > 100:
        # Longer texts might need normalization
        compound_score *= 1.1 if abs(compound_score) > 0.3 else 1.0
    
    # Adjust for question marks (questions often seem less negative)
    question_count = text.count('?')
    if question_count > 0 and compound_score < -0.2:
        compound_score += 0.1 * min(question_count, 3)
    
    # Adjust for exclamation marks
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        if compound_score > 0.2:
            compound_score += 0.1
        elif compound_score < -0.2:
            compound_score -= 0.1
    
    # Cap compound score to valid range
    compound_score = max(-1.0, min(1.0, compound_score))

    sent_pipe, tox_pipe = get_pipelines()

    try:
        if sent_pipe is not None:
            sent_result = sent_pipe(text_for_models)[0]
            sent_scores = {item['label'].lower(): float(item['score']) for item in sent_result}
            pos = sent_scores.get('positive', 0.0)
            neg = sent_scores.get('negative', 0.0)
            neu = sent_scores.get('neutral', 0.0)
        else:
            # Fallback if pipeline not loaded
            pos = max(0.0, compound_score) if compound_score > 0.1 else 0.0
            neg = max(0.0, -compound_score) if compound_score < -0.1 else 0.0
            neu = max(0.0, 1.0 - pos - neg)
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        pos = max(0.0, compound_score) if compound_score > 0.1 else 0.0
        neg = max(0.0, -compound_score) if compound_score < -0.1 else 0.0
        neu = max(0.0, 1.0 - pos - neg)

    try:
        if tox_pipe is not None:
            tox_result = tox_pipe(text_for_models)[0]
            tox_scores = {item['label'].lower(): float(item['score']) for item in tox_result}
            toxic_score = tox_scores.get('toxic', tox_scores.get('toxic/other', 0.0)) or 0.0
        else:
            # Fallback if pipeline not loaded
            toxic_score = 0.3 if rude_keyword_count > 0 or aggressive_pattern_count > 0 else 0.0
    except Exception as e:
        print(f"Error in toxicity analysis: {e}")
        toxic_score = 0.3 if rude_keyword_count > 0 or aggressive_pattern_count > 0 else 0.0

    keyword_rude_boost = min(0.3, rude_keyword_count * 0.15)
    aggressive_boost = min(0.4, aggressive_pattern_count * 0.2)
    if hey_used_rudely:
        aggressive_boost = max(aggressive_boost, 0.35)
    
    keyword_friendly_boost = min(0.2, friendly_keyword_count * 0.1)
    keyword_formal_boost = min(0.2, formal_keyword_count * 0.1)

    rude_score = max(toxic_score, neg) + keyword_rude_boost + aggressive_boost
    if (toxic_score >= 0.4 or 
        (neg > 0.55 and compound_score < -0.2) or 
        (rude_keyword_count > 0 and neg > 0.4) or
        aggressive_pattern_count > 0 or
        hey_used_rudely or
        (rude_keyword_count > 0 and aggressive_pattern_count > 0)):
        tone = 'rude'
        confidence = min(0.98, 0.55 + rude_score * 0.4)
    elif not hey_used_rudely and ((pos >= max(neg, neu) and pos > 0.4) or (friendly_keyword_count > 0 and pos > 0.35) or compound_score > 0.2):
        tone = 'friendly'
        friendly_score = pos + keyword_friendly_boost + max(0, compound_score * 0.5)
        confidence = min(0.95, 0.5 + friendly_score * 0.4)
    else:
        tone = 'formal'
        formal_score = neu + keyword_formal_boost + (1.0 - abs(compound_score) * 0.5)
        confidence = max(0.6, 0.5 + formal_score * 0.3)

    return tone, round(confidence, 3), {'compound': compound_score, 'toxic': toxic_score, 'pos': pos, 'neg': neg, 'neu': neu}

def generate_gpt_suggestions(original_text, model_type='rule-based'):
    """Use different models to generate polite rewrites for rude emails.
    
    Args:
        original_text: The text to rewrite
        model_type: One of 'rule-based', 'openrouter', 'gemini'
    """
    # Extract names from original text to preserve them
    name_pattern = r'\b([A-Z][a-z]+)\b'
    potential_names = re.findall(name_pattern, original_text)
    names_to_preserve = [name for name in potential_names if name not in ['Hi', 'Hello', 'Dear', 'Regards', 'Thanks', 'Best', 'Sincerely']]
    
    # Prepare enhanced prompt that generates complete professional emails
    prompt = f"""You are an expert professional email writer. Transform the following rude or unprofessional email into TWO complete, professional email versions.

CRITICAL REQUIREMENTS:
1. Generate COMPLETE emails with proper greeting, body, and closing
2. PRESERVE all names mentioned in the original email exactly as they appear
3. Create a completely new professional email - do NOT just edit or cut sentences
4. Maintain the core message and intent but express it professionally
5. Remove ALL profanity, insults, offensive language, and aggressive commands
6. Replace rude phrases with polite, constructive alternatives
7. Add appropriate greetings (Hi/Hello/Dear + name if mentioned)
8. Add professional closings (Best regards, Sincerely, etc.)
9. Make the email sound natural and professional

Original Email:
{original_text}

Generate two complete email versions:
1. Friendly version: Warm, approachable, but still professional
2. Formal version: More formal, business-appropriate tone

Return ONLY valid JSON in this exact format (no other text):
{{
  "friendly": "Complete friendly email with greeting, body, and closing",
  "formal": "Complete formal email with greeting, body, and closing"
}}"""

    # Use Gemini
    if model_type == 'gemini':
        api_key = get_gemini_api_key()
        if not api_key:
            return None
        
        endpoint = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }
        
        response = requests.post(endpoint, headers=headers, data=json.dumps(body))
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group(0))
            if result.get('friendly') and result.get('formal'):
                return result
        return None
    
    # Use OpenRouter
    elif model_type == 'openrouter':
        api_key = get_openrouter_api_key()
        if not api_key:
            return None
        
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(endpoint, headers=headers, data=json.dumps(body))
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group(0))
            if result.get('friendly') and result.get('formal'):
                return result
        return None
    
    # Rule-based (fallback) - return None to use rule-based rewrite
    elif model_type == 'rule-based':
        return None
    
    return None

def suggest_polite_rewrite(original_text, tone, model_type='rule-based'):
    """Suggest polite alternatives using selected model for rude emails, fallback to rule-based.
    
    Args:
        original_text: The text to rewrite
        tone: Detected tone (rude, formal, friendly)
        model_type: One of 'rule-based', 'openrouter', 'gemini'
    """
    text = (original_text or '').strip()
    if not text:
        return [{
            'original': original_text,
            'rewritten': original_text,
            'change': 'No content provided'
        }]

    suggestions = []
    
    # Use AI model if not rule-based
    if model_type != 'rule-based':
        gpt_result = generate_gpt_suggestions(text, model_type)
        
        if gpt_result:
            friendly = gpt_result.get('friendly', '').strip()
            formal = gpt_result.get('formal', '').strip()
            
            model_labels = {
                'gemini': 'Gemini',
                'openrouter': 'OpenRouter'
            }
            model_label = model_labels.get(model_type, 'AI')
            
            if friendly and len(friendly) > 20:
                suggestions.append({
                    'original': original_text,
                    'rewritten': friendly,
                    'change': f'{model_label} - Friendly'
                })
            if formal and len(formal) > 20:
                suggestions.append({
                    'original': original_text,
                    'rewritten': formal,
                    'change': f'{model_label} - Formal'
                })
            
            if suggestions:
                return suggestions
    
    # Fallback to rule-based for all cases
    rule_suggestions = create_rule_based_rewrites(original_text, tone)
    if rule_suggestions:
        return rule_suggestions
    
    # Final fallback
    redacted_body = 'I have some concerns regarding parts of the message and would like to discuss constructive next steps.'
    friendly_fb = 'Hi,\n\n' + redacted_body + '\n\nCould we align on next steps?\n\nThanks,'
    formal_fb = 'Hello,\n\n' + redacted_body + '\n\nPlease let me know how we can move forward.\n\nRegards,'
    suggestions.append({'original': original_text, 'rewritten': friendly_fb, 'change': 'Friendly fallback (redacted)'})
    suggestions.append({'original': original_text, 'rewritten': formal_fb, 'change': 'Formal fallback (redacted)'})
    return suggestions

def remove_offensive_words(text, tox_pipe):
    """Use toxicity pipeline to identify and remove offensive words/phrases."""
    if not tox_pipe:
        return text
    
    def check_toxicity(text_to_check, threshold=0.15):
        try:
            text_safe = truncate_text_for_model(text_to_check, max_chars=300)
            res = tox_pipe(text_safe)[0]
            scores = {item['label'].lower(): float(item['score']) for item in res}
            tox_score = 0.0
            for key in ['toxic', 'toxicity', 'abusive', 'obscene']:
                if key in scores:
                    tox_score = max(tox_score, scores[key])
            return tox_score >= threshold
        except Exception:
            return False
    
    words = text.split()
    cleaned_words = []
    skip_next = False
    
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
            
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if not word_clean:
            cleaned_words.append(word)
            continue
        
        if check_toxicity(word_clean, threshold=0.15):
            continue
        
        if i < len(words) - 1:
            next_word_clean = re.sub(r'[^\w]', '', words[i+1].lower())
            phrase = word_clean + ' ' + next_word_clean
            if check_toxicity(phrase, threshold=0.15):
                skip_next = True
                continue
        
        cleaned_words.append(word)
    
    result = ' '.join(cleaned_words)
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def create_rule_based_rewrites(text, tone):
    """Create context-aware rewrites using rule-based transformations that generate complete emails."""
    suggestions = []
    text_lower = text.lower()
    
    # Extract names from text to preserve them
    name_pattern = r'\b([A-Z][a-z]+)\b'
    potential_names = re.findall(name_pattern, text)
    names = [name for name in potential_names if name not in ['Hi', 'Hello', 'Dear', 'Regards', 'Thanks', 'Best', 'Sincerely', 'The', 'This', 'That', 'What', 'When', 'Where', 'Why', 'How', 'I', 'You', 'We', 'They', 'He', 'She', 'It']]
    recipient_name = names[0] if names else None
    
    _, tox_pipe = get_pipelines()
    
    rude_replacements = {
        r'\bstupid\b': 'not ideal',
        r'\bidiot\b': '',
        r'\bidot\b': '',
        r'\bfool\b': 'unwise',
        r'\bcrazy\b': 'unexpected',
        r'\bhorrible\b': 'concerning',
        r'\bterrible\b': 'needs improvement',
        r'\bhate\b': 'prefer not to',
        r'\bdamn\b': '',
        r'\bhell\b': '',
        r'\bwhat the\b': 'I am surprised that',
        r'\bridiculous\b': 'unexpected',
        r'\babsurd\b': 'unusual',
        r'\bpathetic\b': 'disappointing',
        r'\buseless\b': 'needs revision',
        r'\bcrap\b': 'suboptimal',
        r'\bsucks\b': 'needs work',
    }
    
    command_replacements = {
        r'\basap\b': 'at your earliest convenience',
        r'\bsend\s+asap\b': 'please send when you have a chance',
        r'\bnow\s+(send|do|give)\b': r'please \1',
        r'\bgive\s+me\b': 'could you please send me',
        r'\bgive\b': 'please send',
    }
    
    if tone == 'rude':
        friendly_text = text
        
        friendly_text = remove_offensive_words(friendly_text, tox_pipe)
        friendly_text = re.sub(r'\b(damn|hell|damned)\b', '', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\b(idiot|idot)\b', '', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\s+', ' ', friendly_text)
        
        for pattern, replacement in command_replacements.items():
            friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
        
        for pattern, replacement in rude_replacements.items():
            if replacement:
                friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
        
        if re.search(r'\bsend\b', friendly_text.lower()) and 'please' not in friendly_text.lower():
            friendly_text = re.sub(r'\b(send)\b', 'please send', friendly_text, flags=re.IGNORECASE, count=1)
        
        friendly_text = re.sub(r'\b(very|extremely|incredibly)\s+(bad|wrong|awful)', 'concerning', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\bis\s+(terrible|horrible|awful|bad)\b', 'needs improvement', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
        
        if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower):
            friendly_text = re.sub(r'\bhey\b', '', friendly_text, flags=re.IGNORECASE)
            friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
        
        if any(word in text_lower for word in ['hate', 'stupid', 'idiot', 'idot', 'terrible', 'horrible', 'ridiculous']):
            if not any(starter in friendly_text.lower()[:50] for starter in ['i wanted', 'i hope', 'i appreciate', 'hi', 'hello']):
                friendly_text = "I wanted to reach out about this. " + friendly_text
        
        # Add proper greeting
        greeting = f"Hi {recipient_name},\n\n" if recipient_name else "Hi,\n\n"
        if not re.match(r'^(hi|hello|dear|greetings)', friendly_text.lower().strip()):
            friendly_text = greeting + friendly_text
        else:
            friendly_text = re.sub(r'^(hey|hi|hello)\s*[,\-]?\s*', greeting, friendly_text, flags=re.IGNORECASE)
        
        # Ensure proper closing
        if not any(word in friendly_text.lower()[-100:] for word in ['thanks', 'thank you', 'regards', 'sincerely', 'best']):
            friendly_text += "\n\nThanks!"
        
        suggestions.append({
            'original': text,
            'rewritten': friendly_text,
            'change': 'Rule-Based - Friendly'
        })
        
        formal_text = text
        formal_text = remove_offensive_words(formal_text, tox_pipe)
        formal_text = re.sub(r'\b(damn|hell|damned)\b', '', formal_text, flags=re.IGNORECASE)
        formal_text = re.sub(r'\b(idiot|idot)\b', '', formal_text, flags=re.IGNORECASE)
        formal_text = re.sub(r'\s+', ' ', formal_text)
        
        for pattern, replacement in command_replacements.items():
            formal_text = re.sub(pattern, replacement, formal_text, flags=re.IGNORECASE)
        
        for pattern, replacement in rude_replacements.items():
            if replacement:
                formal_text = re.sub(pattern, replacement, formal_text, flags=re.IGNORECASE)
        
        if re.search(r'\bsend\b', formal_text.lower()) and 'please' not in formal_text.lower():
            formal_text = re.sub(r'\b(send)\b', 'please send', formal_text, flags=re.IGNORECASE, count=1)
        
        formal_text = re.sub(r'\b(very|extremely|incredibly)\s+(bad|wrong|awful)', 'concerning', formal_text, flags=re.IGNORECASE)
        formal_text = re.sub(r'\s+', ' ', formal_text).strip()
        
        if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower):
            formal_text = re.sub(r'\bhey\b', '', formal_text, flags=re.IGNORECASE)
            formal_text = re.sub(r'\s+', ' ', formal_text).strip()
        
        # Add proper greeting
        greeting = f"Hello {recipient_name},\n\n" if recipient_name else "Hello,\n\n"
        if not re.match(r'^(dear|hello|greetings)', formal_text.lower().strip()):
            formal_text = greeting + formal_text
        else:
            formal_text = re.sub(r'^(hey|hi|hello)\s*[,\-]?\s*', greeting, formal_text, flags=re.IGNORECASE)
        
        # Ensure proper closing
        if 'regards' not in formal_text.lower() and 'sincerely' not in formal_text.lower():
            formal_text += "\n\nBest regards,"
        
        suggestions.append({
            'original': text,
            'rewritten': formal_text,
            'change': 'Rule-Based - Formal'
        })
        
    elif tone == 'formal':
        text_lower_check = text.lower()
        has_rude_content = (
            any(kw in text_lower_check for kw in RUDE_KEYWORDS) or
            any(re.search(pattern, text_lower_check, re.IGNORECASE) for pattern in AGGRESSIVE_PATTERNS) or
            (re.search(r'\bhey\b\s+(idiot|idot|stupid)', text_lower_check, re.IGNORECASE))
        )
        
        if has_rude_content:
            friendly_text = text
            friendly_text = remove_offensive_words(friendly_text, tox_pipe)
            friendly_text = re.sub(r'\b(damn|hell|damned)\b', '', friendly_text, flags=re.IGNORECASE)
            friendly_text = re.sub(r'\b(idiot|idot)\b', '', friendly_text, flags=re.IGNORECASE)
            friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
            
            for pattern, replacement in command_replacements.items():
                friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
            
            if re.search(r'\bsend\b', friendly_text.lower()) and 'please' not in friendly_text.lower():
                friendly_text = re.sub(r'\b(send)\b', 'please send', friendly_text, flags=re.IGNORECASE, count=1)
            
            if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower_check):
                friendly_text = re.sub(r'\bhey\b', '', friendly_text, flags=re.IGNORECASE)
                friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
            
            # Add proper greeting with name
            greeting = f"Hi {recipient_name},\n\n" if recipient_name else "Hi,\n\n"
            if not re.match(r'^(hi|hello|dear)', friendly_text.lower().strip()):
                friendly_text = greeting + friendly_text
            else:
                friendly_text = re.sub(r'^(hey|hi|hello)\s*[,\-]?\s*', greeting, friendly_text, flags=re.IGNORECASE)
            
            if not any(word in friendly_text.lower()[-100:] for word in ['thanks', 'thank you', 'regards', 'sincerely']):
                friendly_text += "\n\nThanks!"
            
            suggestions.append({
                'original': text,
                'rewritten': friendly_text,
                'change': 'Polite rewrite (removed offensive language)'
            })
        else:
            warm_text = text
            if re.match(r'^(dear|hello)', warm_text.lower()):
                warm_text = re.sub(r'^Dear\s+', 'Hi ', warm_text, flags=re.IGNORECASE)
                warm_text = re.sub(r'^Hello\s*,\s*', 'Hi,\n\n', warm_text, flags=re.IGNORECASE)
            
            if 'sincerely' in warm_text.lower() or 'respectfully' in warm_text.lower():
                warm_text = re.sub(r'(?i)\s*(sincerely|respectfully|yours sincerely)[,.]?\s*$', '\n\nBest,', warm_text)
            
            suggestions.append({
                'original': text,
                'rewritten': warm_text,
                'change': 'Warmer version (maintains professionalism)'
            })
        
    else:
        professional_text = text
        if 'cheers' in professional_text.lower() and 'regards' not in professional_text.lower():
            professional_text = re.sub(r'(?i)\s*cheers\s*[,.]?\s*$', '\n\nBest regards,', professional_text)
        
        suggestions.append({
            'original': text,
            'rewritten': professional_text,
            'change': 'Maintains friendly tone with professionalism'
        })
    
    return suggestions

# ... (rest of the routes remain the same)

@app.route('/')
def index():
    """Main dashboard route"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        import traceback
        traceback.print_exc()
        return f"<h1>Email Tone Checker</h1><p>Service is running. Template error: {str(e)}</p>", 200

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        email_text = data.get('email_text', '')
        model_type = data.get('model_type', 'rule-based')  # Get model selection
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        try:
            tone, confidence, vader_scores = classify_tone(email_text)
        except Exception as e:
            print(f"Error in classify_tone: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error analyzing tone: {str(e)}'}), 500
        
        suggestions = suggest_polite_rewrite(email_text, tone, model_type)
        
        clean_suggestions = []
        for s in (suggestions or []):
            rew = s.get('rewritten', '') or ''
            rew = re.sub(r"'{0,2}---VERSION_SEPARATOR---'{0,2}", '', rew, flags=re.IGNORECASE)
            rew = rew.replace('---VERSION_SEPARATOR---', '')
            rew = rew.replace('\r', '')
            rew = rew.strip()
            if not rew:
                base = (email_text or '').strip()
                if base:
                    rew = 'Hi,\n\n' + base + '\n\nThanks,'
                else:
                    rew = base
            s['rewritten'] = rew
            clean_suggestions.append(s)
        suggestions = clean_suggestions
        
        recommendations = []
        if tone == 'rude':
            recommendations.append('Consider using softer language to maintain professionalism')
            recommendations.append('Avoid negative words that might offend the recipient')
            recommendations.append('Focus on solutions rather than problems')
        elif tone == 'formal':
            recommendations.append('Tone is appropriate for professional communication')
            recommendations.append('Consider adding a friendly greeting if appropriate')
        else:
            recommendations.append('Friendly tone is great for maintaining relationships')
            recommendations.append('Maintain professionalism while being warm')
        
        # Ensure all values are JSON serializable
        serializable_vader_scores = {
            k: float(v) if isinstance(v, (int, float)) else v 
            for k, v in vader_scores.items()
        }
        
        return jsonify({
            'tone': tone,
            'confidence': round(confidence, 2),
            'vader_scores': serializable_vader_scores,
            'suggestions': suggestions,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Unexpected error in analyze_email: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/send_email', methods=['POST'])
def send_email():
    """Send email using SMTP"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        recipient_email = data.get('recipient_email', '').strip()
        email_text = data.get('email_text', '').strip()
        sender_email = data.get('sender_email', '').strip()
        sender_password = data.get('sender_password', '').strip()
        subject = data.get('subject', 'Email from ToneChecker').strip()
        
        if not recipient_email or not email_text:
            return jsonify({'error': 'Recipient email and message text are required'}), 400
        
        if not sender_email or not sender_password:
            return jsonify({'error': 'Sender email and password are required'}), 400
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, recipient_email):
            return jsonify({'error': 'Invalid recipient email format'}), 400
        if not re.match(email_pattern, sender_email):
            return jsonify({'error': 'Invalid sender email format'}), 400
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(email_text, 'plain'))
        
        # Determine SMTP server based on sender email
        if 'gmail.com' in sender_email.lower():
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587
        elif 'outlook.com' in sender_email.lower() or 'hotmail.com' in sender_email.lower():
            smtp_server = 'smtp-mail.outlook.com'
            smtp_port = 587
        elif 'yahoo.com' in sender_email.lower():
            smtp_server = 'smtp.mail.yahoo.com'
            smtp_port = 587
        else:
            # Default to Gmail settings, user can configure custom SMTP
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return jsonify({
            'success': True,
            'message': f'Email sent successfully to {recipient_email}'
        })
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Authentication failed. Please check your email and password, or use an App Password.'}), 401
    except smtplib.SMTPRecipientsRefused:
        return jsonify({'error': 'Recipient email address was refused by the server.'}), 400
    except smtplib.SMTPServerDisconnected:
        return jsonify({'error': 'Connection to email server was lost.'}), 500
    except Exception as e:
        print(f"Error in send_email: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to send email: {str(e)}'}), 500

@app.route('/fetch_emails', methods=['POST'])
def fetch_emails():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        email_address = data.get('email')
        password = data.get('password')
        
        if not email_address or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        mail.login(email_address, password)
        mail.select('inbox')
        
        try:
            limit = int(data.get('limit', 20))
        except Exception:
            limit = 20

        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split() if status == 'OK' else []
        if not email_ids:
            status, messages = mail.search(None, 'ALL')
            email_ids = messages[0].split() if status == 'OK' else []

        emails = []
        for email_id in email_ids[-limit:]:
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    subject_raw = msg['Subject']
                    subject = ''
                    if subject_raw:
                        decoded_parts = decode_header(subject_raw)
                        for part, enc in decoded_parts:
                            if isinstance(part, bytes):
                                try:
                                    subject += part.decode(enc or 'utf-8', errors='ignore')
                                except Exception:
                                    subject += part.decode('utf-8', errors='ignore')
                            else:
                                subject += part
                    if not subject:
                        subject = 'No Subject'
                    
                    sender = msg['From']
                    
                    body = ''
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get('Content-Disposition') or '')
                            if content_type == "text/plain" and 'attachment' not in content_disposition:
                                try:
                                    payload = part.get_payload(decode=True)
                                    charset = part.get_content_charset() or 'utf-8'
                                    body = (payload or b'').decode(charset, errors='ignore')
                                    if body:
                                        break
                                except Exception:
                                    continue
                    else:
                        if msg.get_content_type() == "text/plain":
                            payload = msg.get_payload(decode=True)
                            try:
                                body = (payload or b'').decode(msg.get_content_charset() or 'utf-8', errors='ignore')
                            except Exception:
                                body = (payload or b'').decode('utf-8', errors='ignore')
                    
                    try:
                        tone, confidence, _ = classify_tone(body)
                    except Exception as e:
                        body_lower = body.lower()
                        if any(kw in body_lower for kw in RUDE_KEYWORDS):
                            tone = 'rude'
                            confidence = 0.7
                        elif any(kw in body_lower for kw in FRIENDLY_KEYWORDS):
                            tone = 'friendly'
                            confidence = 0.6
                        else:
                            tone = 'formal'
                            confidence = 0.5
                    
                    emails.append({
                        'id': email_id.decode(),
                        'subject': subject,
                        'sender': sender,
                        'body': body[:200] + '...' if len(body) > 200 else body,
                        'full_body': body,
                        'tone': tone,
                        'confidence': round(confidence, 2)
                    })
        
        mail.close()
        mail.logout()
        
        return jsonify({'emails': emails})
        
    except imaplib.IMAP4.error as e:
        print(f"IMAP error in fetch_emails: {e}")
        return jsonify({'error': 'Invalid credentials. Please enable "Less secure app access" or use an App Password.'}), 401
    except Exception as e:
        print(f"Error in fetch_emails: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'email-tone-checker',
            'transformers_available': TRANSFORMERS_AVAILABLE
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port, use_reloader=False)