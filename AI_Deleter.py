import customtkinter as ctk
from tkinter import messagebox, filedialog, Text
import pyperclip
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import json
import os
from datetime import datetime
import tkinter as tk

class ColoredTextWidget(ctk.CTkFrame):
    """Custom widget that combines CTkFrame with tkinter Text for colored text support"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create tkinter Text widget for colored text
        self.text_widget = Text(
            self,
            wrap="word",
            font=("Segoe UI", 12),
            bg="#343638",  # Dark theme background
            fg="#DCE4EE",  # Light text color
            insertbackground="white",  # Cursor color
            selectbackground="#3B8ED0",  # Selection background
            selectforeground="white",  # Selection text color
            relief="flat",
            borderwidth=0
        )
        
        # Configure tags for highlighting
        self.text_widget.tag_config("paraphraser", foreground="#FF6B6B", background="#2C2C2C")  # Red color
        self.text_widget.tag_config("highlight", background="#2C3E50")  # Blue highlight
        
        self.text_widget.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ctk.CTkScrollbar(self, command=self.text_widget.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_widget.configure(yscrollcommand=scrollbar.set)
    
    def insert(self, index, text, tags=None):
        """Insert text with optional tags"""
        if tags:
            self.text_widget.insert(index, text, tags)
        else:
            self.text_widget.insert(index, text)
    
    def delete(self, start, end=None):
        """Delete text"""
        if end:
            self.text_widget.delete(start, end)
        else:
            self.text_widget.delete(start)
    
    def get(self, start, end=None):
        """Get text"""
        if end:
            return self.text_widget.get(start, end)
        return self.text_widget.get(start)
    
    def configure(self, **kwargs):
        """Configure widget properties"""
        if "height" in kwargs:
            self.text_widget.configure(height=kwargs["height"])
            del kwargs["height"]
        if "font" in kwargs:
            self.text_widget.configure(font=kwargs["font"])
            del kwargs["font"]
        if "wrap" in kwargs:
            self.text_widget.configure(wrap=kwargs["wrap"])
            del kwargs["wrap"]
        super().configure(**kwargs)

class TextHumanizerApp:
    def __init__(self):
        # Initialize the main window
        self.root = ctk.CTk()
        self.root.title("AI Text Humanizer")
        self.root.geometry("1000x900")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Model settings
        self.model_name = "Ateeqq/Text-Rewriter-Paraphraser"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # User settings (with defaults)
        self.settings = {
            "custom_passes": 1,
            "remove_dashes": True,
            "use_custom_passes": False,
            "strength": "High",
            "save_intermediate": True,
            "intermediate_format": "json",
            "highlight_paraphraser": True,  # New: highlight paraphraser word
            "highlight_intensity": 50  # New: highlight intensity (0-100)
        }
        
        # Track intermediate outputs
        self.intermediate_outputs = []
        
        # Settings window reference
        self.settings_window = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = ctk.CTkLabel(
            self.root, 
            text="AI Text Humanizer",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)
        
        # Settings Frame
        settings_frame = ctk.CTkFrame(self.root)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        # Device info
        device_label = ctk.CTkLabel(
            settings_frame, 
            text=f"Device: {self.device.upper()}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="green" if self.device == "cuda" else "orange"
        )
        device_label.pack(side="left", padx=10)
        
        # Model status
        self.model_status = ctk.CTkLabel(
            settings_frame,
            text="Model: Not Loaded",
            text_color="orange"
        )
        self.model_status.pack(side="left", padx=10)
        
        # Load model button
        self.load_model_btn = ctk.CTkButton(
            settings_frame,
            text="Load Model",
            command=self.load_model_thread,
            width=150
        )
        self.load_model_btn.pack(side="left", padx=10)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            settings_frame,
            text="Ready - Load model to begin",
            text_color="orange"
        )
        self.status_label.pack(side="left", padx=10)
        
        # Settings button
        settings_btn = ctk.CTkButton(
            settings_frame,
            text="⚙️ Settings",
            command=self.open_settings,
            width=120
        )
        settings_btn.pack(side="right", padx=10)
        
        # Intermediate outputs button (new)
        self.intermediate_btn = ctk.CTkButton(
            settings_frame,
            text="📁 Save Intermediate",
            command=self.save_intermediate_outputs,
            width=140,
            state="disabled"
        )
        self.intermediate_btn.pack(side="right", padx=5)
        
        # Progress frame
        progress_frame = ctk.CTkFrame(self.root)
        progress_frame.pack(fill="x", padx=20, pady=5)
        
        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Progress: Waiting...",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(side="left", padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(side="left", padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Highlight frame
        highlight_frame = ctk.CTkFrame(self.root)
        highlight_frame.pack(fill="x", padx=20, pady=5)
        
        # Highlight toggle
        self.highlight_toggle = ctk.CTkSwitch(
            highlight_frame,
            text="Highlight 'paraphraser' word",
            command=self.toggle_highlight,
            onvalue=True,
            offvalue=False
        )
        if self.settings["highlight_paraphraser"]:
            self.highlight_toggle.select()
        self.highlight_toggle.pack(side="left", padx=10)
        
        # Highlight intensity slider
        intensity_label = ctk.CTkLabel(highlight_frame, text="Intensity:")
        intensity_label.pack(side="left", padx=(20, 5))
        
        self.intensity_slider = ctk.CTkSlider(
            highlight_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            width=150,
            command=self.update_highlight_intensity
        )
        self.intensity_slider.set(self.settings["highlight_intensity"])
        self.intensity_slider.pack(side="left", padx=5)
        
        self.intensity_value = ctk.CTkLabel(
            highlight_frame,
            text=f"{self.settings['highlight_intensity']}%",
            width=40
        )
        self.intensity_value.pack(side="left", padx=5)
        
        # Options frame
        options_frame = ctk.CTkFrame(self.root)
        options_frame.pack(fill="x", padx=20, pady=5)
        
        # Output selection
        output_label = ctk.CTkLabel(options_frame, text="Select Version:")
        output_label.pack(side="left", padx=10)
        
        self.output_selector = ctk.CTkComboBox(
            options_frame,
            values=["Version 1 (Best)", "Version 2", "Version 3", "Version 4"],
            width=150,
            command=self.change_output
        )
        self.output_selector.set("Version 1 (Best)")
        self.output_selector.pack(side="left", padx=5)
        
        # Humanization strength
        strength_label = ctk.CTkLabel(options_frame, text="Strength:")
        strength_label.pack(side="left", padx=(20, 5))
        
        self.strength_selector = ctk.CTkSegmentedButton(
            options_frame,
            values=["Standard", "High", "Maximum"],
            width=250,
            command=self.update_strength
        )
        self.strength_selector.set(self.settings["strength"])
        self.strength_selector.pack(side="left", padx=5)
        
        # Main content frame
        content_frame = ctk.CTkFrame(self.root)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Input section
        input_label = ctk.CTkLabel(
            content_frame,
            text="Input Text (Paste your entire text here - any length):",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        input_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.input_text = ctk.CTkTextbox(
            content_frame,
            height=250,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.input_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Control buttons
        button_frame = ctk.CTkFrame(content_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        self.humanize_btn = ctk.CTkButton(
            button_frame,
            text="🤖 Humanize Entire Text",
            command=self.humanize_text,
            width=220,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.humanize_btn.pack(side="left", padx=5)
        
        # Info label for passes (updated to use settings)
        self.passes_info = ctk.CTkLabel(
            button_frame,
            text=self.get_passes_info_text(),
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.passes_info.pack(side="left", padx=5)
        
        clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=self.clear_all,
            width=120,
            height=40
        )
        clear_btn.pack(side="left", padx=5)
        
        # Output section
        output_label = ctk.CTkLabel(
            content_frame,
            text="Humanized Output:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        output_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)
        
        # Use custom text widget for output that supports colored text
        self.output_text = ColoredTextWidget(
            content_frame,
            height=250
        )
        self.output_text.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
        
        # Output info label
        self.output_info_label = ctk.CTkLabel(
            content_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="orange"
        )
        self.output_info_label.grid(row=5, column=0, pady=(5, 0))
        
        # Copy button
        copy_btn = ctk.CTkButton(
            content_frame,
            text="📋 Copy to Clipboard",
            command=self.copy_output,
            width=200
        )
        copy_btn.grid(row=6, column=0, pady=10)
        
        # Configure grid weights
        content_frame.grid_rowconfigure(1, weight=1)
        content_frame.grid_rowconfigure(4, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Store multiple outputs
        self.all_outputs = []
        
    def get_passes_info_text(self):
        """Get the text for passes info label"""
        if self.settings["use_custom_passes"]:
            return f"(Custom: {self.settings['custom_passes']} passes)"
        else:
            return f"(Using Strength setting: {self.settings['strength']})"
        
    def update_passes_info(self):
        """Update the passes info label"""
        self.passes_info.configure(text=self.get_passes_info_text())
        
    def load_model_thread(self):
        """Start model loading in separate thread"""
        self.load_model_btn.configure(state="disabled", text="Loading...")
        self.status_label.configure(text="Loading model... This may take a minute", text_color="orange")
        
        thread = threading.Thread(target=self._load_model)
        thread.daemon = True
        thread.start()
    
    def _load_model(self):
        """Load the model in background thread"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Update UI in main thread
            self.root.after(0, self._model_loaded_success)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, self._model_loaded_error, error_msg)
    
    def _model_loaded_success(self):
        """Update UI after successful model load"""
        self.model_status.configure(
            text="Model: Loaded ✓",
            text_color="green"
        )
        self.status_label.configure(
            text=f"Ready - Using {self.device.upper()}",
            text_color="green"
        )
        self.load_model_btn.configure(state="disabled", text="Model Loaded")
        self.humanize_btn.configure(state="normal")
        self.intermediate_btn.configure(state="normal")
        
        messagebox.showinfo(
            "Success", 
            f"Model loaded successfully!\nUsing: {self.device.upper()}\n\nYou can now paste entire texts of any length.\nThe tool will automatically split and process them in optimal chunks."
        )
    
    def _model_loaded_error(self, error_msg):
        """Update UI after model load error"""
        self.status_label.configure(
            text="✗ Failed to load model",
            text_color="red"
        )
        self.load_model_btn.configure(state="normal", text="Retry Load")
        
        messagebox.showerror(
            "Error",
            f"Failed to load model:\n\n{error_msg}\n\nMake sure you have internet connection and enough disk space."
        )
    
    def split_into_chunks(self, text):
        """Split text into chunks that fit within token limit"""
        # First split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            # Count tokens for this sentence
            tokens = self.tokenizer.encode(f"paraphraser: {sentence}", add_special_tokens=True)
            num_tokens = len(tokens)
            
            # If single sentence is too long, split it
            if num_tokens > 60:  # Increased limit
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by commas or semicolons
                parts = re.split(r'[,;]\s+', sentence)
                for part in parts:
                    part_tokens = len(self.tokenizer.encode(f"paraphraser: {part}", add_special_tokens=True))
                    if part_tokens > 60:
                        # Just add as is if still too long
                        chunks.append(part)
                    else:
                        chunks.append(part)
            else:
                # Check if adding this sentence exceeds limit
                if current_tokens + num_tokens > 60:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = num_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += num_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def humanize_output(self, text):
        """Post-process text to make it more human-like"""
        result = text
        
        # Remove AI-typical long dashes and replace with commas
        if self.settings["remove_dashes"]:
            # Replace em-dash (—) and en-dash (–) with comma
            result = re.sub(r'\s*[—–]\s*', ', ', result)
            
            # Also handle cases where dashes are used without spaces
            result = re.sub(r'[—–]', ', ', result)
            
            # Clean up any double commas
            result = re.sub(r',\s*,', ',', result)
            
            # Fix spacing around commas
            result = re.sub(r'\s*,\s*', ', ', result)
            
            # Remove comma before period
            result = re.sub(r',\s*\.', '.', result)
        
        # Remove other AI patterns (optional enhancements)
        # Remove multiple spaces
        result = re.sub(r'\s+', ' ', result)
        
        # Trim
        result = result.strip()
        
        return result
    
    def update_strength(self, value):
        """Update current strength setting"""
        self.settings["strength"] = value
        self.update_passes_info()
    
    def toggle_highlight(self):
        """Toggle paraphraser highlighting"""
        self.settings["highlight_paraphraser"] = self.highlight_toggle.get()
        # Re-apply highlighting to current output
        if self.all_outputs:
            self.apply_highlighting(self.output_text.get("1.0", "end"))
    
    def update_highlight_intensity(self, value):
        """Update highlight intensity"""
        intensity = int(float(value))
        self.settings["highlight_intensity"] = intensity
        self.intensity_value.configure(text=f"{intensity}%")
        # Update highlight color based on intensity
        self.update_highlight_color()
        # Re-apply highlighting if enabled
        if self.settings["highlight_paraphraser"] and self.all_outputs:
            self.apply_highlighting(self.output_text.get("1.0", "end"))
    
    def update_highlight_color(self):
        """Update highlight color based on intensity"""
        intensity = self.settings["highlight_intensity"]
        # Calculate color based on intensity (0% = light pink, 100% = bright red)
        red_intensity = int(255 * (intensity / 100))
        color_hex = f"#{min(255, red_intensity):02X}{max(0, 100 - red_intensity):02X}{max(0, 100 - red_intensity):02X}"
        
        # Update the tag configuration
        self.output_text.text_widget.tag_config("paraphraser", foreground=color_hex, background="#2C2C2C")
    
    def apply_highlighting(self, text):
        """Apply highlighting to paraphraser words in text"""
        # Clear current text
        self.output_text.delete("1.0", "end")
        
        if not self.settings["highlight_paraphraser"]:
            # Just insert text without highlighting
            self.output_text.insert("1.0", text)
            return
        
        # Split text and apply highlighting
        pattern = r'\b(paraphraser)\b'
        parts = re.split(f'({pattern})', text, flags=re.IGNORECASE)
        
        # Count occurrences
        paraphraser_count = 0
        
        # Insert parts with appropriate tags
        for part in parts:
            if re.match(pattern, part, re.IGNORECASE):
                self.output_text.insert("end", part, "paraphraser")
                paraphraser_count += 1
            elif part:
                self.output_text.insert("end", part)
        
        # Update info label
        if paraphraser_count > 0:
            self.output_info_label.configure(
                text=f"⚠️ Found {paraphraser_count} 'paraphraser' word(s) that need manual fixing!",
                text_color="#FF6B6B"
            )
        else:
            self.output_info_label.configure(text="")
    
    def toggle_custom_passes(self, entry_widget, switch_widget):
        """Toggle custom passes entry field"""
        is_enabled = switch_widget.get()
        if is_enabled:
            entry_widget.configure(state="normal")
        else:
            entry_widget.configure(state="disabled")
        # Update setting immediately
        self.settings["use_custom_passes"] = is_enabled
        self.update_passes_info()
    
    def open_settings(self):
        """Open settings window"""
        # If settings window is already open, bring it to front
        if hasattr(self, 'settings_window') and self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
            
        self.settings_window = ctk.CTkToplevel(self.root)
        self.settings_window.title("Settings")
        self.settings_window.geometry("600x800")
        self.settings_window.resizable(True, True)
        self.settings_window.transient(self.root)
        
        # Make window modal
        self.settings_window.grab_set()
        
        # Handle window close
        def on_closing():
            self.settings_window.destroy()
            self.settings_window = None
            
        self.settings_window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Title
        title = ctk.CTkLabel(
            self.settings_window,
            text="Humanization Settings",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=20)
        
        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(self.settings_window, height=550)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Use custom passes toggle
        use_custom_label = ctk.CTkLabel(
            scroll_frame,
            text="Override Strength Setting:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        use_custom_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        # Create switch first
        use_custom_switch = ctk.CTkSwitch(
            scroll_frame,
            text="Use Custom Rephrasing Passes",
            onvalue=True,
            offvalue=False
        )
        if self.settings["use_custom_passes"]:
            use_custom_switch.select()
        use_custom_switch.pack(anchor="w", padx=10, pady=5)
        
        use_custom_desc = ctk.CTkLabel(
            scroll_frame,
            text="When OFF: Uses Strength setting (Standard/High/Maximum)\nWhen ON: Uses custom passes below",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        use_custom_desc.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Custom passes setting
        passes_label = ctk.CTkLabel(
            scroll_frame,
            text="Custom Rephrasing Passes:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        passes_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        passes_desc = ctk.CTkLabel(
            scroll_frame,
            text="Number of times to rephrase each chunk (any positive number)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        passes_desc.pack(anchor="w", padx=10, pady=(0, 5))
        
        # Frame for passes entry
        passes_frame = ctk.CTkFrame(scroll_frame)
        passes_frame.pack(anchor="w", padx=10, pady=5, fill="x")
        
        passes_entry = ctk.CTkEntry(
            passes_frame,
            width=150,
            placeholder_text="Enter any number (e.g., 200)"
        )
        passes_entry.insert(0, str(self.settings["custom_passes"]))
        passes_entry.pack(side="left", padx=(0, 10))
        
        # Set initial state of passes entry
        if self.settings["use_custom_passes"]:
            passes_entry.configure(state="normal")
        else:
            passes_entry.configure(state="disabled")
        
        # Dash removal setting
        dash_label = ctk.CTkLabel(
            scroll_frame,
            text="Punctuation Humanization:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        dash_label.pack(anchor="w", padx=10, pady=(20, 5))
        
        dash_desc = ctk.CTkLabel(
            scroll_frame,
            text="Remove AI-typical long dashes (—, –) and replace with commas",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        dash_desc.pack(anchor="w", padx=10, pady=(0, 5))
        
        dash_switch = ctk.CTkSwitch(
            scroll_frame,
            text="Remove long dashes",
            onvalue=True,
            offvalue=False
        )
        if self.settings["remove_dashes"]:
            dash_switch.select()
        dash_switch.pack(anchor="w", padx=10, pady=5)
        
        # NEW: Save intermediate outputs setting
        intermediate_label = ctk.CTkLabel(
            scroll_frame,
            text="Intermediate Outputs:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        intermediate_label.pack(anchor="w", padx=10, pady=(20, 5))
        
        intermediate_desc = ctk.CTkLabel(
            scroll_frame,
            text="Save all intermediate paraphrases (not just final results)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        intermediate_desc.pack(anchor="w", padx=10, pady=(0, 5))
        
        intermediate_switch = ctk.CTkSwitch(
            scroll_frame,
            text="Save intermediate paraphrases",
            onvalue=True,
            offvalue=False
        )
        if self.settings["save_intermediate"]:
            intermediate_switch.select()
        intermediate_switch.pack(anchor="w", padx=10, pady=5)
        
        # Format selection for intermediate outputs
        format_label = ctk.CTkLabel(
            scroll_frame,
            text="Intermediate Format:",
            font=ctk.CTkFont(size=12)
        )
        format_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        format_selector = ctk.CTkSegmentedButton(
            scroll_frame,
            values=["JSON", "TXT"],
            width=150
        )
        format_selector.set(self.settings["intermediate_format"].upper())
        format_selector.pack(anchor="w", padx=10, pady=5)
        
        # NEW: Highlight paraphraser setting
        highlight_label = ctk.CTkLabel(
            scroll_frame,
            text="Paraphraser Detection:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        highlight_label.pack(anchor="w", padx=10, pady=(20, 5))
        
        highlight_desc = ctk.CTkLabel(
            scroll_frame,
            text="Highlight the word 'paraphraser' in red when AI leaves it in output",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        highlight_desc.pack(anchor="w", padx=10, pady=(0, 5))
        
        highlight_switch = ctk.CTkSwitch(
            scroll_frame,
            text="Highlight 'paraphraser' word",
            onvalue=True,
            offvalue=False
        )
        if self.settings["highlight_paraphraser"]:
            highlight_switch.select()
        highlight_switch.pack(anchor="w", padx=10, pady=5)
        
        # Highlight intensity slider in settings
        intensity_label_settings = ctk.CTkLabel(
            scroll_frame,
            text="Highlight Intensity:",
            font=ctk.CTkFont(size=12)
        )
        intensity_label_settings.pack(anchor="w", padx=10, pady=(10, 5))
        
        intensity_slider_settings = ctk.CTkSlider(
            scroll_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            width=200
        )
        intensity_slider_settings.set(self.settings["highlight_intensity"])
        intensity_slider_settings.pack(anchor="w", padx=10, pady=5)
        
        intensity_value_settings = ctk.CTkLabel(
            scroll_frame,
            text=f"{self.settings['highlight_intensity']}%",
            width=40
        )
        intensity_value_settings.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Info section
        info_label = ctk.CTkLabel(
            scroll_frame,
            text="💡 Tips for Best Results:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        info_label.pack(anchor="w", padx=10, pady=(20, 5))
        
        info_text = ctk.CTkLabel(
            scroll_frame,
            text="• Standard Strength: 1 pass, 4 beams\n• High Strength: 1 pass, 8 beams (recommended)\n• Maximum Strength: 2 passes, 10 beams\n• Custom: Any number of passes allowed\n• Higher passes = longer processing time\n• Dash removal makes text more casual\n• Intermediate outputs show progress through passes\n• 'paraphraser' highlighting helps spot AI artifacts",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            justify="left"
        )
        info_text.pack(anchor="w", padx=10, pady=5)
        
        # Configure switch command to update entry state and setting
        def toggle_command():
            self.toggle_custom_passes(passes_entry, use_custom_switch)
            
        use_custom_switch.configure(command=toggle_command)
        
        # Function to update passes setting
        def update_passes(*args):
            try:
                value = passes_entry.get().strip()
                if value and value.isdigit():
                    passes = int(value)
                    if passes > 0:
                        self.settings["custom_passes"] = passes
                        self.update_passes_info()
                    else:
                        messagebox.showwarning("Invalid Input", "Please enter a positive number")
                        passes_entry.delete(0, "end")
                        passes_entry.insert(0, str(self.settings["custom_passes"]))
            except ValueError:
                pass
        
        # Bind entry change
        passes_entry.bind("<KeyRelease>", update_passes)
        
        # Function to update dash setting
        def update_dash(*args):
            self.settings["remove_dashes"] = dash_switch.get()
            
        dash_switch.configure(command=update_dash)
        
        # Function to update intermediate setting
        def update_intermediate(*args):
            self.settings["save_intermediate"] = intermediate_switch.get()
            
        intermediate_switch.configure(command=update_intermediate)
        
        # Function to update format setting
        def update_format(*args):
            self.settings["intermediate_format"] = format_selector.get().lower()
            
        format_selector.configure(command=update_format)
        
        # Function to update highlight setting
        def update_highlight(*args):
            self.settings["highlight_paraphraser"] = highlight_switch.get()
            # Update main UI toggle
            self.highlight_toggle.select() if highlight_switch.get() else self.highlight_toggle.deselect()
            # Re-apply highlighting if needed
            if self.all_outputs:
                current_output = self.output_text.get("1.0", "end")
                self.apply_highlighting(current_output)
            
        highlight_switch.configure(command=update_highlight)
        
        # Function to update highlight intensity in settings
        def update_intensity_settings(value):
            intensity = int(float(value))
            self.settings["highlight_intensity"] = intensity
            intensity_value_settings.configure(text=f"{intensity}%")
            # Update main UI slider
            self.intensity_slider.set(intensity)
            self.intensity_value.configure(text=f"{intensity}%")
            # Update highlight color
            self.update_highlight_color()
            # Re-apply highlighting if enabled
            if self.settings["highlight_paraphraser"] and self.all_outputs:
                current_output = self.output_text.get("1.0", "end")
                self.apply_highlighting(current_output)
            
        intensity_slider_settings.configure(command=update_intensity_settings)
        
        # Close button
        close_btn = ctk.CTkButton(
            self.settings_window,
            text="Close Settings",
            command=on_closing,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        close_btn.pack(pady=20)
    
    def save_intermediate_outputs(self):
        """Save intermediate outputs to file"""
        if not self.intermediate_outputs:
            messagebox.showinfo("No Data", "No intermediate outputs to save. Run humanization first.")
            return
        
        # Ask for save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.settings["intermediate_format"] == "json":
            filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
            default_ext = ".json"
            default_name = f"intermediate_outputs_{timestamp}.json"
        else:
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
            default_ext = ".txt"
            default_name = f"intermediate_outputs_{timestamp}.txt"
        
        filename = filedialog.asksaveasfilename(
            title="Save Intermediate Outputs",
            defaultextension=default_ext,
            filetypes=filetypes,
            initialfile=default_name
        )
        
        if not filename:
            return
        
        try:
            if self.settings["intermediate_format"] == "json":
                # Save as JSON
                output_data = {
                    "timestamp": timestamp,
                    "settings": self.settings,
                    "intermediate_outputs": self.intermediate_outputs,
                    "final_outputs": self.all_outputs
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                    
                messagebox.showinfo("Success", f"Intermediate outputs saved to:\n{filename}")
                
            else:
                # Save as TXT
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Intermediate Outputs - {timestamp}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Write settings
                    f.write("SETTINGS:\n")
                    f.write(f"  Strength: {self.settings['strength']}\n")
                    f.write(f"  Custom Passes: {self.settings['custom_passes']}\n")
                    f.write(f"  Use Custom Passes: {self.settings['use_custom_passes']}\n")
                    f.write(f"  Remove Dashes: {self.settings['remove_dashes']}\n\n")
                    
                    # Write intermediate outputs
                    f.write("INTERMEDIATE OUTPUTS:\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, chunk_data in enumerate(self.intermediate_outputs):
                        f.write(f"CHUNK {i + 1}:\n")
                        f.write("-" * 30 + "\n")
                        
                        for pass_num, outputs in enumerate(chunk_data['outputs']):
                            f.write(f"\n  Pass {pass_num + 1}:\n")
                            for j, output in enumerate(outputs):
                                f.write(f"    Version {j + 1}: {output[:100]}...\n")
                        f.write("\n")
                    
                    # Write final outputs
                    f.write("\nFINAL OUTPUTS:\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, output in enumerate(self.all_outputs):
                        f.write(f"Version {i + 1}:\n")
                        f.write("-" * 30 + "\n")
                        f.write(output + "\n\n")
                
                messagebox.showinfo("Success", f"Intermediate outputs saved to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")
    
    def humanize_text(self):
        """Humanize the input text"""
        input_content = self.input_text.get("1.0", "end-1c").strip()
        
        if not input_content:
            messagebox.showwarning("No Input", "Please enter some text to humanize.")
            return
        
        if self.model is None:
            messagebox.showwarning("Model Not Loaded", "Please load the model first.")
            return
        
        # Disable button and show processing
        self.humanize_btn.configure(state="disabled", text="Processing...")
        self.status_label.configure(text="Splitting text into chunks...", text_color="orange")
        self.output_text.delete("1.0", "end")
        self.progress_bar.set(0)
        self.output_info_label.configure(text="")
        
        # Clear previous intermediate outputs
        self.intermediate_outputs = []
        
        # Run in separate thread
        thread = threading.Thread(target=self._process_text, args=(input_content,))
        thread.daemon = True
        thread.start()
    
    def _process_text(self, input_content):
        """Process text in background thread"""
        try:
            # Get humanization strength for beam parameters
            strength = self.settings["strength"]
            
            # Determine passes: use custom if enabled, otherwise use strength-based
            if self.settings["use_custom_passes"]:
                passes = self.settings["custom_passes"]
            else:
                passes = {"Standard": 1, "High": 1, "Maximum": 2}[strength]
            
            # Adjust beam parameters based on strength
            if strength == "Standard":
                num_beams = 4
                repetition_penalty = 10.0
            elif strength == "High":
                num_beams = 8
                repetition_penalty = 12.0
            else:  # Maximum
                num_beams = 10
                repetition_penalty = 15.0
            
            # Split text into optimal chunks
            chunks = self.split_into_chunks(input_content)
            total_chunks = len(chunks) * passes
            
            self.root.after(0, self._update_progress, 0, total_chunks, 
                          f"Processing with {passes} pass{'es' if passes != 1 else ''}...")
            
            # Process each chunk and collect 4 versions
            all_versions = [[], [], [], []]  # 4 different versions
            
            chunk_counter = 0
            for chunk_index, chunk in enumerate(chunks):
                # Initialize chunk data for intermediate outputs
                if self.settings["save_intermediate"]:
                    chunk_data = {
                        'original': chunk,
                        'outputs': []  # Will store outputs for each pass
                    }
                
                # Process chunk (potentially multiple times for Maximum strength)
                processed_chunk = chunk
                
                for pass_num in range(passes):
                    # Update progress
                    self.root.after(0, self._update_progress, chunk_counter, total_chunks, 
                                  f"Processing chunk {chunk_counter+1}/{total_chunks}")
                    
                    # Add the required prefix
                    prefixed_text = f"paraphraser: {processed_chunk}"
                    
                    # Tokenize
                    input_ids = self.tokenizer(
                        prefixed_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=80
                    ).input_ids.to(self.device)
                    
                    # Generate with aggressive humanization parameters
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            num_beams=num_beams,
                            num_return_sequences=4,
                            repetition_penalty=repetition_penalty,
                            length_penalty=1.5,
                            no_repeat_ngram_size=3,
                            max_length=80,
                            min_length=10,
                            early_stopping=True,
                            do_sample=False
                        )
                    
                    # Decode the 4 different outputs
                    paraphrases = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    # Apply post-processing humanization
                    paraphrases = [self.humanize_output(p) for p in paraphrases]
                    
                    # Save intermediate outputs if enabled
                    if self.settings["save_intermediate"]:
                        chunk_data['outputs'].append(paraphrases.copy())
                    
                    # For multiple passes, use first output as input for next pass
                    if pass_num < passes - 1:
                        processed_chunk = paraphrases[0]
                    
                    chunk_counter += 1
                
                # Add chunk data to intermediate outputs
                if self.settings["save_intermediate"]:
                    self.intermediate_outputs.append(chunk_data)
                
                # Add final results to each version
                for i, paraphrase in enumerate(paraphrases):
                    all_versions[i].append(paraphrase)
            
            # Join chunks for each version
            self.all_outputs = [' '.join(version) for version in all_versions]
            
            # Update final progress
            self.root.after(0, self._update_progress, total_chunks, total_chunks, "Complete!")
            
            # Update UI with first version
            self.root.after(0, self._update_output, self.all_outputs[0], True)
            
        except Exception as e:
            error_msg = f"Error during processing:\n{str(e)}"
            self.root.after(0, self._update_output, error_msg, False)
    
    def _update_progress(self, current, total, message):
        """Update progress bar and label"""
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)
        self.progress_label.configure(text=f"Progress: {message}")
    
    def change_output(self, choice):
        """Change displayed output based on selection"""
        if not self.all_outputs:
            return
        
        # Extract version number
        version_num = int(choice.split()[1]) - 1
        
        # Get the selected output
        output_text = self.all_outputs[version_num]
        
        # Apply highlighting
        self.apply_highlighting(output_text)
        
        self.status_label.configure(
            text=f"Showing {choice}",
            text_color="green"
        )
    
    def _update_output(self, text, success):
        """Update output text box (called from main thread)"""
        # Apply highlighting
        self.apply_highlighting(text)
        
        self.humanize_btn.configure(state="normal", text="🤖 Humanize Entire Text")
        
        if success:
            dash_msg = " | Dashes removed" if self.settings["remove_dashes"] else ""
            intermediate_msg = " | Intermediate saved" if self.settings["save_intermediate"] else ""
            highlight_msg = " | Highlight active" if self.settings["highlight_paraphraser"] else ""
            self.status_label.configure(
                text=f"✓ Complete! Try different versions{dash_msg}{intermediate_msg}{highlight_msg}",
                text_color="green"
            )
            self.progress_label.configure(text="Progress: Done! ✓")
            
            # Enable intermediate button if we have intermediate outputs
            if self.settings["save_intermediate"] and self.intermediate_outputs:
                self.intermediate_btn.configure(state="normal")
        else:
            self.status_label.configure(
                text="✗ Processing failed",
                text_color="red"
            )
            self.progress_bar.set(0)
            self.progress_label.configure(text="Progress: Failed")
    
    def copy_output(self):
        """Copy output text to clipboard"""
        output_content = self.output_text.get("1.0", "end-1c").strip()
        
        if not output_content:
            messagebox.showwarning("No Output", "No text to copy.")
            return
        
        try:
            pyperclip.copy(output_content)
            self.status_label.configure(
                text="✓ Copied to clipboard",
                text_color="green"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy: {str(e)}")
    
    def clear_all(self):
        """Clear all text boxes"""
        self.input_text.delete("1.0", "end")
        self.output_text.delete("1.0", "end")
        self.all_outputs = []
        self.intermediate_outputs = []
        self.progress_bar.set(0)
        self.progress_label.configure(text="Progress: Waiting...")
        self.status_label.configure(
            text=f"Ready - Using {self.device.upper()}" if self.model else "Ready - Load model to begin",
            text_color="green" if self.model else "orange"
        )
        self.output_info_label.configure(text="")
        self.intermediate_btn.configure(state="disabled" if not self.model else "normal")
    
    def run(self):
        """Start the application"""
        # Initialize highlight color
        self.update_highlight_color()
        self.root.mainloop()


if __name__ == "__main__":
    app = TextHumanizerApp()
    app.run()
