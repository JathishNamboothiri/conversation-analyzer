
"""
*************************    DCONVAN   *******************************
Deep Conversation Analysis Streamlit App
Features emotion flow graphs and sentiment distribution analysis
Uses advanced AI Conversation Analyzer Engine for detailed conversation analysis

*************************    DCONVAN   *******************************
"""
 
import streamlit as st
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
import statistics
from collections import defaultdict, Counter
import re
import io

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import openai
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error("Missing required packages. Please install: streamlit openai pandas plotly")
    st.stop()

class EnhancedConversationAnalyzer:
    """
    Enhanced conversation analyzer with emotion flow and sentiment distribution
    """
    
    def __init__(self):
        """Initialize the analyzer with API key from environment variables"""
        # Get API key from environment variables (secure for deployment)
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            st.error("⚠️ API key not found. Please set OPENAI_API_KEY in your environment variables.")
            st.info("For local testing, create a .streamlit/secrets.toml file with: OPENAI_API_KEY = 'your-key-here'")
            st.stop()
        
        # Initialize AI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Emotion categories for tracking
        self.emotion_categories = [
            "joy", "sadness", "anger", "fear", "surprise", "disgust", 
            "trust", "anticipation", "neutral", "frustration", "satisfaction"
        ]
        
        # Sentiment categories
        self.sentiment_categories = ["positive", "negative", "neutral"]
        
        # Analysis categories
        self.analysis_categories = {
            "sentiment": "Overall emotional tone and sentiment",
            "discrimination": "Caste, religion, race, gender-based discrimination",
            "offensive_language": "Abusive, profane, or offensive content",
            "politeness": "Politeness, respect, and conversational civility", 
            "aggression": "Aggressive, hostile, or confrontational behavior",
            "conversation_quality": "Overall conversation dynamics and quality",
            "bias_detection": "Implicit and explicit biases",
            "power_dynamics": "Conversational dominance and power structures"
        }
        
    def load_transcript(self, uploaded_file):
        """Load transcript from uploaded file"""
        try:
            if uploaded_file is not None:
                # Read the uploaded file
                file_contents = uploaded_file.read()
                data = json.loads(file_contents)
                
                if not data.get('success'):
                    raise ValueError("Transcript file indicates processing failure")
                
                # Extract key components
                transcript_data = {
                    'segments': data.get('segments', []),
                    'speakers': data.get('transcription_info', {}).get('speakers_detected', []),
                    'metadata': data.get('file_info', {}),
                    'full_transcript': data.get('full_transcript', ''),
                    'processing_info': data.get('processing_info', {}),
                    'transcription_info': data.get('transcription_info', {})
                }
                
                return transcript_data
            return None
            
        except Exception as e:
            st.error(f"Error loading transcript: {e}")
            return None
    
    def prepare_conversation_context(self, transcript_data):
        """Prepare conversation context for analysis"""
        segments = transcript_data['segments']
        
        # Organize by speaker
        speaker_segments = defaultdict(list)
        conversation_flow = []
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').strip()
            
            if text:
                speaker_segments[speaker].append({
                    'text': text,
                    'start_time': segment.get('start_time', 0),
                    'end_time': segment.get('end_time', 0),
                    'duration': segment.get('duration', 0)
                })
                
                conversation_flow.append({
                    'speaker': speaker,
                    'text': text,
                    'timestamp': segment.get('start_time', 0),
                    'segment_index': len(conversation_flow)
                })
        
        # Sort conversation flow by time
        conversation_flow.sort(key=lambda x: x['timestamp'])
        
        # Calculate speaker statistics
        speaker_stats = {}
        for speaker, speaker_segs in speaker_segments.items():
            total_words = sum(len(seg['text'].split()) for seg in speaker_segs)
            total_duration = sum(seg['duration'] for seg in speaker_segs)
            
            speaker_stats[speaker] = {
                'total_segments': len(speaker_segs),
                'total_words': total_words,
                'total_speaking_time': total_duration,
                'average_segment_length': total_duration / len(speaker_segs) if speaker_segs else 0,
                'words_per_minute': (total_words / (total_duration / 60)) if total_duration > 0 else 0
            }
        
        return {
            'speaker_segments': dict(speaker_segments),
            'conversation_flow': conversation_flow,
            'speaker_stats': speaker_stats,
            'total_speakers': len(speaker_segments)
        }
    
    def create_enhanced_analysis_prompt(self, conversation_context, transcript_data):
        """Create enhanced analysis prompt with emotion flow tracking"""
        
        full_conversation = transcript_data['full_transcript']
        speakers = conversation_context['speaker_stats']
        conversation_flow = conversation_context['conversation_flow']
        
        # Create timeline for emotion tracking
        timeline_segments = []
        for i, segment in enumerate(conversation_flow):
            timeline_segments.append(f"[{segment['timestamp']:.1f}s] {segment['speaker']}: {segment['text']}")
        
        timeline_text = "\n".join(timeline_segments)
        
        prompt = f"""
You are an expert conversation analyst specializing in emotion flow tracking and comprehensive behavioral analysis. 

## CONVERSATION METADATA:
- Duration: {transcript_data['metadata'].get('duration_formatted', 'unknown')}
- Total Speakers: {conversation_context['total_speakers']}
- Total Segments: {len(transcript_data['segments'])}

## SPEAKER STATISTICS:
{json.dumps(speakers, indent=2)}

## FULL CONVERSATION TRANSCRIPT:
{full_conversation}

## CONVERSATION TIMELINE:
{timeline_text}

## ANALYSIS REQUIREMENTS:

Please provide a comprehensive analysis covering the following aspects:

### 1. EMOTION FLOW ANALYSIS
For each speaker, track emotions throughout the conversation with timestamps:
- Map emotions to specific time periods
- Identify emotion transitions and triggers
- Track emotional intensity levels (1-10)

### 2. SENTIMENT DISTRIBUTION
Analyze overall sentiment patterns:
- Positive, negative, neutral percentages per speaker
- Sentiment changes over time
- Conversation sentiment arc

### 3. COMPREHENSIVE BEHAVIORAL ANALYSIS
- Overall conversation sentiment (positive/negative/neutral)
- Individual speaker sentiment profiles
- Sentiment progression throughout the conversation
- Emotional turning points and triggers
- Mood shifts and their causes
- Religious discrimination or bias
- Caste-based discrimination or references
- Racial discrimination or stereotyping
- Gender-based discrimination or bias
- Cultural or ethnic bias
- Profanity or vulgar language usage
- Hate speech or derogatory terms
- Insults or personal attacks
- Threatening language
- Level of politeness and courtesy
- Respectful vs disrespectful interactions
- Aggressive language patterns
- Hostile or confrontational behavior
- Speaker dominance patterns
- Power imbalances in conversation

## OUTPUT FORMAT:

Provide your analysis as a structured JSON response with the following format:

{{
  "emotion_flow": {{
    "SPEAKER_00": [
      {{
        "timestamp": 0.23,
        "emotion": "neutral",
        "intensity": 5,
        "trigger": "conversation start",
        "text_segment": "for ultra-fast conversation."
      }},
      {{
        "timestamp": 28.62,
        "emotion": "frustration",
        "intensity": 7,
        "trigger": "perceived impoliteness",
        "text_segment": "Why are you being so impolite to me?"
      }}
    ],
    "SPEAKER_01": [
      {{
        "timestamp": 3.38,
        "emotion": "curiosity",
        "intensity": 6,
        "trigger": "greeting attempt",
        "text_segment": "Hey, what is your name?"
      }}
    ]
  }},
  "sentiment_distribution": {{
    "overall": {{
      "positive": 30.0,
      "negative": 45.0,
      "neutral": 25.0
    }},
    "by_speaker": {{
      "SPEAKER_00": {{
        "positive": 35.0,
        "negative": 40.0,
        "neutral": 25.0,
        "dominant_sentiment": "negative"
      }},
      "SPEAKER_01": {{
        "positive": 25.0,
        "negative": 50.0,
        "neutral": 25.0,
        "dominant_sentiment": "negative"
      }}
    }},
    "temporal_sentiment": [
      {{
        "time_range": "0-30s",
        "sentiment": "neutral",
        "score": 0.1
      }},
      {{
        "time_range": "30-60s",
        "sentiment": "negative",
        "score": -0.6
      }},
      {{
        "time_range": "60-90s",
        "sentiment": "negative",
        "score": -0.8
      }},
      {{
        "time_range": "90-120s",
        "sentiment": "negative",
        "score": -0.5
      }},
      {{
        "time_range": "120-180s",
        "sentiment": "positive",
        "score": 0.4
      }}
    ]
  }},
  "overall_assessment": {{
    "conversation_type": "customer service interaction with conflict",
    "dominant_sentiment": "negative",
    "severity_level": "moderate",
    "primary_concerns": ["discrimination", "offensive language", "aggression"],
    "conversation_quality_score": 4,
    "emotional_volatility": 7
  }},
  "speaker_profiles": {{
    "SPEAKER_00": {{
      "sentiment_profile": "Initially neutral, becomes defensive",
      "dominant_emotions": ["neutral", "defensive", "apologetic"],
      "emotional_stability": 6,
      "behavioral_patterns": ["apologetic responses", "professional demeanor"],
      "communication_style": "formal and professional",
      "concerning_behaviors": [],
      "politeness_score": 8,
      "aggression_score": 1
    }},
    "SPEAKER_01": {{
      "sentiment_profile": "Frustrated and confrontational",
      "dominant_emotions": ["frustration", "anger", "satisfaction"],
      "emotional_stability": 4,
      "behavioral_patterns": ["impatience", "discriminatory comments", "eventual satisfaction"],
      "communication_style": "informal and direct",
      "concerning_behaviors": ["gender discrimination", "insulting language"],
      "politeness_score": 3,
      "aggression_score": 7
    }}
  }},
  "discrimination_analysis": {{
    "detected": true,
    "types": ["gender", "religious"],
    "severity": "moderate",
    "specific_instances": [
      "I don't think women's are suited for such leadership roles.",
      "People from your religion can't be trusted with money."
    ],
    "affected_groups": ["women", "religious minorities"],
    "timestamps": [53.37, 71.9]
  }},
  "offensive_content": {{
    "detected": true,
    "severity": "moderate",
    "types": ["insults"],
    "instances": [
      "You are completely useless, and this service is a joke."
    ],
    "frequency": "low",
    "timestamps": [93.76]
  }},
  "conversation_health": {{
    "respectful_communication": false,
    "constructive_dialogue": true,
    "conflict_present": true,
    "resolution_attempted": true,
    "overall_toxicity": 6,
    "recommendations": [
      "Avoid discriminatory language",
      "Focus on constructive dialogue",
      "Practice patience in service interactions",
      "Implement bias awareness training"
    ]
  }},
  "temporal_analysis": {{
    "conversation_phases": [
      {{
        "phase": "introduction",
        "start_time": 0.0,
        "end_time": 30.0,
        "description": "Initial interaction and confusion",
        "dominant_emotion": "neutral",
        "sentiment": "neutral"
      }},
      {{
        "phase": "conflict_escalation",
        "start_time": 30.0,
        "end_time": 100.0,
        "description": "Customer frustration and discriminatory comments",
        "dominant_emotion": "anger",
        "sentiment": "negative"
      }},
      {{
        "phase": "resolution",
        "start_time": 100.0,
        "end_time": 180.0,
        "description": "Constructive dialogue and satisfaction",
        "dominant_emotion": "satisfaction",
        "sentiment": "positive"
      }}
    ],
    "escalation_points": [
      {{
        "timestamp": 28.62,
        "trigger": "perceived impoliteness",
        "severity": "moderate"
      }},
      {{
        "timestamp": 53.37,
        "trigger": "discriminatory comment about women",
        "severity": "high"
      }},
      {{
        "timestamp": 71.9,
        "trigger": "religious discrimination",
        "severity": "high"
      }},
      {{
        "timestamp": 93.76,
        "trigger": "insulting language",
        "severity": "moderate"
      }}
    ],
    "emotional_peaks": [
      {{
        "timestamp": 53.37,
        "emotion": "bias",
        "intensity": 8,
        "speaker": "SPEAKER_01"
      }},
      {{
        "timestamp": 93.76,
        "emotion": "frustration",
        "intensity": 9,
        "speaker": "SPEAKER_01"
      }},
      {{
        "timestamp": 158.18,
        "emotion": "satisfaction",
        "intensity": 7,
        "speaker": "SPEAKER_01"
      }}
    ]
  }},
  "detailed_findings": {{
    "positive_aspects": [
      "Professional response to discrimination",
      "Conflict resolution achieved",
      "Polite conclusion"
    ],
    "concerning_aspects": [
      "Gender-based discrimination",
      "Religious prejudice",
      "Insulting language",
      "Customer aggression"
    ],
    "improvement_suggestions": [
      "Implement bias awareness training",
      "Develop de-escalation techniques",
      "Create inclusive communication guidelines",
      "Monitor for discriminatory language"
    ],
    "risk_assessment": "moderate"
  }}
}}

## ANALYSIS GUIDELINES:
- Be objective and evidence-based
- Cite specific quotes when identifying issues
- Consider cultural context and nuance
- Distinguish between intent and impact
- Provide actionable insights
- Be thorough but concise
- Use severity scales consistently
- Consider power dynamics and context

Provide only the JSON response without additional explanation or formatting.
"""
        
        return prompt
    
    def analyze_with_conversation_engine(self, prompt):
        """Send analysis prompt to Conversation Analyzer Engine"""
        try:
            with st.spinner("Analyzing conversation with AI engine..."):
                response = self.client.chat.completions.create(
                    model="gpt-4.1-mini",  # Valid model name
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert conversation analyst specializing in emotion flow tracking, sentiment analysis, and behavioral assessment. Always respond with valid JSON format including detailed emotion flow and sentiment distribution data."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=4000,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                analysis_text = response.choices[0].message.content
                analysis_result = json.loads(analysis_text)
                
                return analysis_result
                
        except json.JSONDecodeError as e:
            st.error(f"Error parsing AI response: {e}")
            return None
            
        except Exception as e:
            st.error(f"Error calling Conversation Analyzer Engine: {e}")
            return None
    
    def generate_statistical_analysis(self, transcript_data, conversation_context):
        """Generate statistical analysis of conversation patterns"""
        segments = transcript_data['segments']
        speaker_stats = conversation_context['speaker_stats']
        
        # Basic statistics
        total_duration = transcript_data['metadata'].get('duration_seconds', 0)
        total_words = transcript_data['transcription_info'].get('total_words', 0)
        
        # Speaking time distribution
        speaking_time_dist = {}
        for speaker, stats in speaker_stats.items():
            speaking_time_dist[speaker] = {
                'percentage': (stats['total_speaking_time'] / total_duration * 100) if total_duration > 0 else 0,
                'words_percentage': (stats['total_words'] / total_words * 100) if total_words > 0 else 0
            }
        
        # Segment length analysis
        segment_lengths = [seg.get('duration', 0) for seg in segments]
        avg_segment_length = statistics.mean(segment_lengths) if segment_lengths else 0
        median_segment_length = statistics.median(segment_lengths) if segment_lengths else 0
        
        # Turn-taking analysis
        speaker_changes = 0
        prev_speaker = None
        longest_turn = 0
        current_turn_duration = 0
        
        for seg in segments:
            current_speaker = seg.get('speaker')
            duration = seg.get('duration', 0)
            
            if prev_speaker and current_speaker != prev_speaker:
                speaker_changes += 1
                longest_turn = max(longest_turn, current_turn_duration)
                current_turn_duration = duration
            else:
                current_turn_duration += duration
            
            prev_speaker = current_speaker
        
        # Final turn
        longest_turn = max(longest_turn, current_turn_duration)
        
        return {
            'basic_stats': {
                'total_duration_seconds': total_duration,
                'total_words': total_words,
                'average_words_per_minute': total_words / (total_duration / 60) if total_duration > 0 else 0,
                'total_segments': len(segments),
                'average_segment_length': avg_segment_length,
                'median_segment_length': median_segment_length
            },
            'speaker_distribution': speaking_time_dist,
            'turn_taking': {
                'speaker_changes': speaker_changes,
                'average_turn_length': total_duration / speaker_changes if speaker_changes > 0 else total_duration,
                'longest_turn_seconds': longest_turn,
                'turn_frequency': speaker_changes / (total_duration / 60) if total_duration > 0 else 0
            }
        }
    
    def create_emotion_flow_chart(self, emotion_flow_data):
        """Create interactive emotion flow chart"""
        fig = go.Figure()
        
        colors = {
            'joy': '#FFD700', 'sadness': '#4169E1', 'anger': '#DC143C',
            'fear': '#800080', 'surprise': '#FF6347', 'disgust': '#228B22',
            'trust': '#20B2AA', 'anticipation': '#FF69B4', 'neutral': '#808080',
            'frustration': '#8B0000', 'satisfaction': '#32CD32'
        }
        
        for speaker, emotions in emotion_flow_data.items():
            if emotions:  # Check if emotions list is not empty
                timestamps = [e.get('timestamp', 0) for e in emotions]
                emotion_values = []
                hover_texts = []
                
                # Convert emotions to numeric values for plotting
                emotion_to_num = {emotion: i for i, emotion in enumerate(self.emotion_categories)}
                
                for e in emotions:
                    emotion = e.get('emotion', 'neutral')
                    intensity = e.get('intensity', 5)
                    emotion_values.append(emotion_to_num.get(emotion, 8) + intensity/10)
                    hover_texts.append(f"Time: {e.get('timestamp', 0):.1f}s<br>"
                                     f"Emotion: {emotion}<br>"
                                     f"Intensity: {intensity}<br>"
                                     f"Text: {e.get('text_segment', '')[:50]}...")
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=emotion_values,
                    mode='lines+markers',
                    name=speaker,
                    hovertext=hover_texts,
                    hoverinfo='text',
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title="Emotion Flow Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Emotion Categories",
            yaxis=dict(
                tickvals=list(range(len(self.emotion_categories))),
                ticktext=self.emotion_categories
            ),
            hovermode='closest',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_sentiment_distribution_chart(self, sentiment_data):
        """Create sentiment distribution charts"""
        # Overall sentiment pie chart
        overall = sentiment_data.get('overall', {})
        
        fig1 = go.Figure(data=[go.Pie(
            labels=['Positive', 'Negative', 'Neutral'],
            values=[overall.get('positive', 0), overall.get('negative', 0), overall.get('neutral', 0)],
            hole=0.3,
            marker_colors=['#28a745', '#dc3545', '#6c757d']
        )])
        fig1.update_layout(title="Overall Sentiment Distribution", height=400)
        
        # Speaker sentiment comparison
        by_speaker = sentiment_data.get('by_speaker', {})
        if by_speaker:
            speakers = list(by_speaker.keys())
            positive_vals = [by_speaker[s].get('positive', 0) for s in speakers]
            negative_vals = [by_speaker[s].get('negative', 0) for s in speakers]
            neutral_vals = [by_speaker[s].get('neutral', 0) for s in speakers]
            
            fig2 = go.Figure(data=[
                go.Bar(name='Positive', x=speakers, y=positive_vals, marker_color='#28a745'),
                go.Bar(name='Negative', x=speakers, y=negative_vals, marker_color='#dc3545'),
                go.Bar(name='Neutral', x=speakers, y=neutral_vals, marker_color='#6c757d')
            ])
            fig2.update_layout(
                title="Sentiment Distribution by Speaker",
                barmode='stack',
                yaxis_title="Percentage",
                height=400
            )
        else:
            fig2 = go.Figure()
            fig2.update_layout(title="No speaker sentiment data available")
        
        # Temporal sentiment
        temporal = sentiment_data.get('temporal_sentiment', [])
        if temporal:
            time_ranges = [t.get('time_range', '') for t in temporal]
            scores = [t.get('score', 0) for t in temporal]
            
            fig3 = go.Figure(data=go.Scatter(
                x=time_ranges,
                y=scores,
                mode='lines+markers',
                marker_color='#17a2b8',
                line=dict(width=3)
            ))
            fig3.update_layout(
                title="Sentiment Over Time",
                xaxis_title="Time Range",
                yaxis_title="Sentiment Score",
                height=400
            )
        else:
            fig3 = go.Figure()
            fig3.update_layout(title="No temporal sentiment data available")
        
        return fig1, fig2, fig3
    
    def create_speaker_comparison_chart(self, speaker_profiles):
        """Create speaker comparison radar chart"""
        if not speaker_profiles:
            return go.Figure().update_layout(title="No speaker profile data available")
        
        categories = ['Politeness', 'Emotional Stability', 'Communication Quality', 
                     'Respectfulness', 'Engagement']
        
        fig = go.Figure()
        
        for speaker, profile in speaker_profiles.items():
            values = [
                profile.get('politeness_score', 5),
                profile.get('emotional_stability', 5),
                5,  # Placeholder for communication quality
                10 - profile.get('aggression_score', 5),  # Invert aggression for respectfulness
                5   # Placeholder for engagement
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=speaker
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Speaker Profile Comparison",
            height=500
        )
        
        return fig
    
    def create_analysis_summary(self, gpt_analysis, statistical_analysis):
        """Create executive summary of analysis results"""
        summary = {
            'key_findings': [],
            'risk_level': 'low',
            'action_required': False,
            'primary_concerns': [],
            'positive_aspects': []
        }
        
        try:
            # Extract key information from AI analysis
            overall = gpt_analysis.get('overall_assessment', {})
            discrimination = gpt_analysis.get('discrimination_analysis', {})
            offensive = gpt_analysis.get('offensive_content', {})
            health = gpt_analysis.get('conversation_health', {})
            
            # Determine risk level
            risk_factors = []
            
            if discrimination.get('detected') and discrimination.get('severity') in ['moderate', 'severe']:
                risk_factors.append('discrimination')
                summary['primary_concerns'].append('Discrimination detected')
            
            if offensive.get('detected') and offensive.get('severity') in ['moderate', 'severe']:
                risk_factors.append('offensive_content')
                summary['primary_concerns'].append('Offensive language present')
            
            if health.get('overall_toxicity', 0) > 6:
                risk_factors.append('high_toxicity')
                summary['primary_concerns'].append('High conversation toxicity')
            
            # Set risk level
            if len(risk_factors) >= 2 or any(s in ['severe', 'critical'] for s in [discrimination.get('severity'), offensive.get('severity')]):
                summary['risk_level'] = 'high'
                summary['action_required'] = True
            elif len(risk_factors) >= 1:
                summary['risk_level'] = 'moderate'
                summary['action_required'] = True
            
            # Identify positive aspects
            if health.get('respectful_communication'):
                summary['positive_aspects'].append('Respectful communication maintained')
            
            if health.get('constructive_dialogue'):
                summary['positive_aspects'].append('Constructive dialogue present')
            
            if overall.get('conversation_quality_score', 0) > 7:
                summary['positive_aspects'].append('High conversation quality')
            
            # Key findings
            summary['key_findings'] = [
                f"Conversation type: {overall.get('conversation_type', 'unknown')}",
                f"Dominant sentiment: {overall.get('dominant_sentiment', 'unknown')}",
                f"Speakers detected: {len(statistical_analysis.get('speaker_distribution', {}))}",
                f"Total duration: {statistical_analysis['basic_stats']['total_duration_seconds']/60:.1f} minutes"
            ]
            
        except Exception as e:
            st.warning(f"Summary generation had issues: {e}")
        
        return summary

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Enhanced Conversation Analyzer",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🗣️ Enhanced Conversation Analysis App")
    st.markdown("*Analyze conversations with AI-powered emotion flow and sentiment distribution*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.success("✅ AI Engine Ready")
        
        # File upload
        st.header("📁 Upload Transcript")
        uploaded_file = st.file_uploader(
            "Choose a transcript JSON file",
            type=['json'],
            help="Upload the JSON transcript file from your ASR engine"
        )
        
        # Clear previous results when new file is uploaded
        if uploaded_file:
            current_file_name = uploaded_file.name
            if 'current_file' not in st.session_state or st.session_state['current_file'] != current_file_name:
                # Clear all previous session data when new file is uploaded
                for key in list(st.session_state.keys()):
                    if key != 'current_file':
                        del st.session_state[key]
                st.session_state['current_file'] = current_file_name
                st.rerun()
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Add clear button
            if st.button("🗑️ Clear Analysis", help="Clear current analysis results"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main content area
    if uploaded_file:
        # Initialize analyzer
        try:
            analyzer = EnhancedConversationAnalyzer()
        except ValueError as e:
            st.error(f"Configuration error: {e}")
            st.stop()
        
        # Load transcript
        with st.spinner("📖 Loading transcript..."):
            transcript_data = analyzer.load_transcript(uploaded_file)
        
        if not transcript_data:
            st.error("Failed to load transcript data")
            st.stop()
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", transcript_data['metadata'].get('duration_formatted', 'N/A'))
        with col2:
            st.metric("Speakers", len(transcript_data['speakers']))
        with col3:
            st.metric("Segments", len(transcript_data['segments']))
        with col4:
            st.metric("Words", transcript_data['transcription_info'].get('total_words', 0))
        
        # Prepare conversation context
        conversation_context = analyzer.prepare_conversation_context(transcript_data)
        
        # Analysis button
        if st.button("🧠 Analyze Conversation", type="primary"):
            # Create analysis prompt
            prompt = analyzer.create_enhanced_analysis_prompt(conversation_context, transcript_data)
            
            # Get AI analysis
            analysis_result = analyzer.analyze_with_conversation_engine(prompt)
            
            if analysis_result:
                # Generate statistical analysis
                statistical_analysis = analyzer.generate_statistical_analysis(transcript_data, conversation_context)
                
                # Create analysis summary
                analysis_summary = analyzer.create_analysis_summary(analysis_result, statistical_analysis)
                
                # Store in session state
                st.session_state['analysis_result'] = analysis_result
                st.session_state['transcript_data'] = transcript_data
                st.session_state['statistical_analysis'] = statistical_analysis
                st.session_state['analysis_summary'] = analysis_summary
                st.success("✅ Analysis completed!")
            else:
                st.error("❌ Analysis failed. Please try again.")
        
        # Display results if available
        if 'analysis_result' in st.session_state:
            analysis_result = st.session_state['analysis_result']
            statistical_analysis = st.session_state.get('statistical_analysis', {})
            analysis_summary = st.session_state.get('analysis_summary', {})
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", "😊 Emotion Flow", "💭 Sentiment Analysis", 
                "👥 Speaker Profiles", "⚠️ Issues & Recommendations"
            ])
            
            with tab1:
                st.subheader("📋 Analysis Overview")
                
                overall = analysis_result.get('overall_assessment', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Conversation Quality", f"{overall.get('conversation_quality_score', 'N/A')}/10")
                    st.metric("Emotional Volatility", f"{overall.get('emotional_volatility', 'N/A')}/10")
                    st.write(f"**Conversation Type:** {overall.get('conversation_type', 'Unknown')}")
                    st.write(f"**Dominant Sentiment:** {overall.get('dominant_sentiment', 'Unknown')}")
                
                with col2:
                    if analysis_summary.get('primary_concerns'):
                        st.error("🚨 **Primary Concerns:**")
                        for concern in analysis_summary['primary_concerns']:
                            st.write(f"• {concern}")
                    else:
                        st.success("✅ No major concerns detected")
                    
                    st.write(f"**Risk Level:** {analysis_summary.get('risk_level', 'unknown').upper()}")
                    st.write(f"**Action Required:** {'YES' if analysis_summary.get('action_required') else 'NO'}")
            
            with tab2:
                st.subheader("😊 Emotion Flow Analysis")
                
                emotion_flow = analysis_result.get('emotion_flow', {})
                if emotion_flow:
                    fig = analyzer.create_emotion_flow_chart(emotion_flow)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Emotional peaks
                    temporal = analysis_result.get('temporal_analysis', {})
                    emotional_peaks = temporal.get('emotional_peaks', [])
                    if emotional_peaks:
                        st.subheader("🎯 Emotional Peaks")
                        for peak in emotional_peaks:
                            st.write(f"**{peak.get('timestamp', 0):.1f}s** - {peak.get('speaker', 'Unknown')}: "
                                   f"{peak.get('emotion', 'unknown')} (intensity: {peak.get('intensity', 0)})")
                else:
                    st.warning("No emotion flow data available")
            
            with tab3:
                st.subheader("💭 Sentiment Distribution")
                
                sentiment_data = analysis_result.get('sentiment_distribution', {})
                if sentiment_data:
                    fig1, fig2, fig3 = analyzer.create_sentiment_distribution_chart(sentiment_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.warning("No sentiment distribution data available")
            
            with tab4:
                st.subheader("👥 Speaker Profile Analysis")
                
                speaker_profiles = analysis_result.get('speaker_profiles', {})
                if speaker_profiles:
                    # Radar chart
                    fig = analyzer.create_speaker_comparison_chart(speaker_profiles)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed profiles
                    for speaker, profile in speaker_profiles.items():
                        with st.expander(f"📊 {speaker} Profile"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Communication Style:** {profile.get('communication_style', 'N/A')}")
                                st.write(f"**Dominant Emotions:** {', '.join(profile.get('dominant_emotions', []))}")
                                st.write(f"**Sentiment Profile:** {profile.get('sentiment_profile', 'N/A')}")
                            with col2:
                                st.metric("Politeness Score", f"{profile.get('politeness_score', 'N/A')}/10")
                                st.metric("Emotional Stability", f"{profile.get('emotional_stability', 'N/A')}/10")
                                st.metric("Aggression Score", f"{profile.get('aggression_score', 'N/A')}/10")
                else:
                    st.warning("No speaker profile data available")
            
            with tab5:
                st.subheader("⚠️ Issues & Recommendations")
                
                # Discrimination analysis
                discrimination = analysis_result.get('discrimination_analysis', {})
                if discrimination.get('detected'):
                    st.error("🚨 **Discrimination Detected**")
                    st.write(f"**Types:** {', '.join(discrimination.get('types', []))}")
                    st.write(f"**Severity:** {discrimination.get('severity', 'Unknown')}")
                    
                    if discrimination.get('specific_instances'):
                        st.write("**Specific Instances:**")
                        for instance in discrimination['specific_instances']:
                            st.write(f"• {instance}")
                
                # Offensive content
                offensive = analysis_result.get('offensive_content', {})
                if offensive.get('detected'):
                    st.error("🤬 **Offensive Content Detected**")
                    st.write(f"**Types:** {', '.join(offensive.get('types', []))}")
                    st.write(f"**Severity:** {offensive.get('severity', 'Unknown')}")
                
                # Recommendations
                health = analysis_result.get('conversation_health', {})
                if health.get('recommendations'):
                    st.success("💡 **Recommendations:**")
                    for rec in health['recommendations']:
                        st.write(f"• {rec}")
                
                # Download analysis report
                if st.button("📥 Download Full Report"):
                    complete_report = {
                        'metadata': {
                            'analysis_timestamp': datetime.now().isoformat(),
                            'analyzer_version': '2.0',
                            'ai_model': 'conversation-analyzer-engine'
                        },
                        'input_data': {
                            'file_info': transcript_data['metadata'],
                            'transcription_info': transcript_data['transcription_info']
                        },
                        'statistical_analysis': statistical_analysis,
                        'ai_analysis': analysis_result,
                        'analysis_summary': analysis_summary
                    }
                    
                    report_json = json.dumps(complete_report, indent=2)
                    st.download_button(
                        label="Download JSON Report",
                        data=report_json,
                        file_name=f"conversation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    else:
        # Instructions when no file is uploaded
        st.info("👋 Welcome! Upload a transcript JSON file to get started.")
        
        with st.expander("📚 How to use this app"):
            st.markdown("""
            1. **Upload a transcript JSON file** from your ASR engine
            2. **Click 'Analyze Conversation'** to start the analysis
            3. **Explore the results** in the different tabs:
               - **Overview**: Key metrics and summary
               - **Emotion Flow**: How emotions change over time
               - **Sentiment Analysis**: Distribution of positive/negative/neutral sentiment
               - **Speaker Profiles**: Individual speaker characteristics
               - **Issues & Recommendations**: Potential problems and suggestions
            """)
        
        with st.expander("🔧 Supported JSON Format"):
            st.markdown("""
            Your transcript JSON should have this structure:
            ```json
            {
              "success": true,
              "segments": [
                {
                  "speaker": "SPEAKER_00",
                  "start_time": 0.23,
                  "end_time": 1.83,
                  "text": "Hello there"
                }
              ],
              "transcription_info": {
                "speakers_detected": ["SPEAKER_00", "SPEAKER_01"]
              }
            }
            ```
            """)

if __name__ == "__main__":
    main()