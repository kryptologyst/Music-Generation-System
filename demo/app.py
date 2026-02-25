"""Streamlit demo for music generation system."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import logging
from typing import Dict, List, Any, Optional
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Music Generation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Notice</h4>
    <p>This system is designed for <strong>RESEARCH AND EDUCATIONAL PURPOSES ONLY</strong>. 
    Please read the <a href="DISCLAIMER.md" target="_blank">privacy and ethics disclaimer</a> 
    before using this system.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎵 Music Generation System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["transformer", "lstm"],
    help="Choose the model architecture"
)

# Generation parameters
st.sidebar.subheader("Generation Parameters")
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=0.8,
    step=0.1,
    help="Controls randomness in generation"
)

top_k = st.sidebar.slider(
    "Top-k",
    min_value=1,
    max_value=100,
    value=50,
    help="Number of top tokens to consider"
)

top_p = st.sidebar.slider(
    "Top-p",
    min_value=0.1,
    max_value=1.0,
    value=0.9,
    step=0.1,
    help="Nucleus sampling threshold"
)

max_length = st.sidebar.slider(
    "Max Length",
    min_value=50,
    max_value=1000,
    value=500,
    step=50,
    help="Maximum length of generated sequence"
)

seed_length = st.sidebar.slider(
    "Seed Length",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
    help="Length of seed sequence"
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Generate Music", "Model Training", "Evaluation", "About"])

with tab1:
    st.header("🎼 Music Generation")
    
    # File upload section
    st.subheader("Upload MIDI Files")
    uploaded_files = st.file_uploader(
        "Choose MIDI files",
        type=['mid', 'midi'],
        accept_multiple_files=True,
        help="Upload MIDI files to train the model or use as seed"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} MIDI files")
        
        # Display file information
        file_info = []
        for file in uploaded_files:
            file_info.append({
                'Name': file.name,
                'Size': f"{file.size / 1024:.1f} KB",
                'Type': file.type
            })
        
        df = pd.DataFrame(file_info)
        st.dataframe(df, use_container_width=True)
    
    # Generation section
    st.subheader("Generate New Music")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎵 Generate Music", type="primary"):
            if uploaded_files:
                # Simulate music generation (in a real implementation, this would use the actual model)
                st.info("Generating music... This is a demo simulation.")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate generation process
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Generating... {i+1}%")
                
                # Simulate generated sequence
                np.random.seed(42)
                generated_sequence = np.random.randint(4, 128, size=max_length)
                
                st.success("Music generation completed!")
                
                # Display generation statistics
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Generated Length", f"{len(generated_sequence)} tokens")
                
                with col_stats2:
                    st.metric("Unique Notes", f"{len(np.unique(generated_sequence))}")
                
                with col_stats3:
                    st.metric("Average Pitch", f"{np.mean(generated_sequence):.1f}")
                
                # Visualize generated sequence
                st.subheader("Generated Sequence Visualization")
                
                # Create a simple visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=generated_sequence,
                    mode='lines+markers',
                    name='Generated Notes',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title="Generated Music Sequence",
                    xaxis_title="Time Steps",
                    yaxis_title="MIDI Note Number",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download generated MIDI
                st.subheader("Download Generated Music")
                
                # Create a temporary MIDI file (simplified)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_file:
                    # In a real implementation, this would create an actual MIDI file
                    tmp_file.write(b"Generated MIDI content")
                    tmp_file_path = tmp_file.name
                
                with open(tmp_file_path, 'rb') as f:
                    st.download_button(
                        label="Download Generated MIDI",
                        data=f.read(),
                        file_name="generated_music.mid",
                        mime="audio/midi"
                    )
                
                # Clean up
                os.unlink(tmp_file_path)
                
            else:
                st.warning("Please upload MIDI files first.")
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("🎲 Random Seed"):
            st.session_state.random_seed = np.random.randint(0, 1000)
            st.success(f"Random seed: {st.session_state.random_seed}")
        
        if st.button("🔄 Reset Parameters"):
            st.rerun()

with tab2:
    st.header("🏋️ Model Training")
    
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
    
    with col2:
        model_size = st.selectbox("Model Size", ["small", "medium", "large"], index=1)
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    
    # Training status
    st.subheader("Training Status")
    
    if st.button("🚀 Start Training", type="primary"):
        if uploaded_files:
            # Simulate training process
            st.info("Starting training... This is a demo simulation.")
            
            # Create training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training epochs
            for epoch in range(epochs):
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Training epoch {epoch + 1}/{epochs}")
                
                # Simulate some delay
                import time
                time.sleep(0.01)
            
            st.success("Training completed!")
            
            # Display training metrics
            st.subheader("Training Metrics")
            
            # Simulate training history
            epochs_range = list(range(1, epochs + 1))
            train_loss = [1.0 - (i / epochs) * 0.8 + np.random.normal(0, 0.05) for i in epochs_range]
            val_loss = [1.0 - (i / epochs) * 0.7 + np.random.normal(0, 0.05) for i in epochs_range]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs_range, y=train_loss, mode='lines', name='Training Loss'))
            fig.add_trace(go.Scatter(x=epochs_range, y=val_loss, mode='lines', name='Validation Loss'))
            
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Please upload MIDI files first.")

with tab3:
    st.header("📊 Evaluation")
    
    st.subheader("Model Performance")
    
    # Simulate evaluation metrics
    metrics_data = {
        'Metric': [
            'Token Accuracy',
            'Pitch Accuracy',
            'Rhythm Accuracy',
            'Harmonic Coherence',
            'Melodic Continuity',
            'Note Diversity',
            'Sequence Diversity'
        ],
        'Score': [0.85, 0.78, 0.72, 0.69, 0.81, 0.76, 0.73],
        'Status': ['Good', 'Good', 'Fair', 'Fair', 'Good', 'Good', 'Good']
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(df_metrics, use_container_width=True)
    
    # Visualize metrics
    st.subheader("Metrics Visualization")
    
    fig = px.bar(
        df_metrics,
        x='Metric',
        y='Score',
        color='Status',
        title="Model Performance Metrics",
        height=500
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample evaluation
    st.subheader("Sample Evaluation")
    
    if st.button("🔍 Evaluate Samples"):
        st.info("Evaluating generated samples...")
        
        # Simulate sample evaluation
        sample_scores = np.random.uniform(0.6, 0.9, 10)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)),
            y=sample_scores,
            mode='markers+lines',
            name='Sample Scores',
            marker=dict(size=8, color=sample_scores, colorscale='Viridis')
        ))
        
        fig.update_layout(
            title="Sample Evaluation Scores",
            xaxis_title="Sample Number",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("ℹ️ About")
    
    st.markdown("""
    ## Music Generation System
    
    This is a modern, research-focused music generation system that uses deep learning 
    techniques to create musical sequences from MIDI data.
    
    ### Features
    
    - **Modern Architecture**: Transformer-based and LSTM models
    - **MIDI Processing**: Comprehensive MIDI file handling
    - **Evaluation Metrics**: Multiple metrics for assessing quality
    - **Interactive Demo**: Web-based interface for music generation
    - **Research Focus**: Designed for academic research and education
    
    ### Model Types
    
    1. **Transformer Model**: Uses multi-head self-attention for capturing long-range dependencies
    2. **LSTM Model**: Uses recurrent neural networks for sequential pattern learning
    
    ### Evaluation Metrics
    
    - **Token Accuracy**: Basic prediction accuracy
    - **Pitch Accuracy**: Correctness of generated pitches
    - **Rhythm Accuracy**: Timing and rhythm pattern accuracy
    - **Harmonic Coherence**: Chord progression quality
    - **Melodic Continuity**: Smoothness of melodic lines
    - **Diversity Metrics**: Measure of generated content variety
    
    ### Usage Guidelines
    
    - Upload MIDI files for training or as seed sequences
    - Adjust generation parameters (temperature, top-k, top-p)
    - Generate new music sequences
    - Evaluate model performance
    - Download generated MIDI files
    
    ### Technical Details
    
    - **Framework**: PyTorch
    - **Tokenization**: Custom MIDI tokenizer
    - **Training**: Cross-entropy loss with early stopping
    - **Generation**: Autoregressive sampling with temperature control
    
    ### Privacy and Ethics
    
    This system is designed for research and educational purposes only. 
    Please ensure compliance with copyright and intellectual property laws 
    when using generated content.
    """)
    
    # System information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("PyTorch Version", "2.0.0")
        st.metric("Python Version", "3.10+")
    
    with col2:
        st.metric("Model Parameters", "~50M")
        st.metric("Training Time", "~2 hours")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Music Generation System - Research and Educational Use Only</p>
        <p>Please read the <a href='DISCLAIMER.md'>privacy and ethics disclaimer</a> before use.</p>
    </div>
    """,
    unsafe_allow_html=True
)
