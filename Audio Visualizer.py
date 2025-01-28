import pygame
from pygame.locals import *
import numpy as np
import pyaudio
import wave
import os
from pydub import AudioSegment
from scipy.fftpack import fft

# Initialize Pygame and constants
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("FFT Visualizer")
clock = pygame.time.Clock()

# Constants for audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Global particles list
particles = []

class Particle:
    def __init__(self, x, y, color, size, velocity):
        self.x = x
        self.y = y
        self.color = color
        self.size = size * 0.5
        self.velocity = velocity
        self.alpha = 255

    def move(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        self.size *= 0.98
        self.alpha -= 2
        if self.alpha <= 0:
            self.alpha = 0

    def draw(self, screen):
        pygame.draw.circle(screen, self.color + (self.alpha,), (int(self.x), int(self.y)), int(self.size))

def get_color(value):
    return (value, int(255 - value), int(128))

# --- FFT to Extract Frequency Information ---
def get_dominant_frequency(data, sample_rate):
    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1 / sample_rate)
    positive_freqs = fft_freq[:len(fft_freq)//2]
    positive_fft = np.abs(fft_result[:len(fft_freq)//2])
    dominant_freq = positive_freqs[np.argmax(positive_fft)]
    return dominant_freq

# --- Map Frequency to Color ---
def map_frequency_to_color(frequency):
    max_freq = 2000  # Maximum frequency range
    min_freq = 20    # Minimum frequency range
    normalized_freq = np.clip((frequency - min_freq) / (max_freq - min_freq), 0, 1)
    hue = normalized_freq * 360  # Map to hue (0-360 range)
    color = pygame.Color(0)
    color.hsva = (hue, 100, 100)  # HSV to RGB conversion
    return color

# --- Smooth the waveform ---
def smooth_waveform(data, window_size=5):
    # Apply a moving average filter to smooth the waveform
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def visualize_fft(data_chunk):
    global particles

    # Perform FFT and get magnitude
    fft_data = fft(data_chunk)
    fft_magnitude = np.abs(fft_data[:len(fft_data) // 2])

    # Normalize FFT values
    max_magnitude = np.max(fft_magnitude)
    if max_magnitude > 0:
        fft_magnitude /= max_magnitude
    else:
        fft_magnitude = np.zeros_like(fft_magnitude)  # If max_magnitude is 0, set all magnitudes to 0

    # Logarithmic scaling for frequency bins
    num_bars = 100
    bar_width = WIDTH // num_bars
    log_bins = np.logspace(0.1, np.log10(len(fft_magnitude)), num=num_bars, dtype=int)
    log_bins = np.unique(log_bins)  # Ensure no duplicate bins

    for i in range(len(log_bins) - 1):
        start_bin = log_bins[i]
        end_bin = log_bins[i + 1]
        intensity = np.mean(fft_magnitude[start_bin:end_bin])  # Average magnitude for this bin

        # Visualize bar
        color = get_color(intensity * 255)
        bar_height = int(intensity * HEIGHT * 0.8)
        x = i * bar_width
        y = HEIGHT - bar_height
        pygame.draw.rect(screen, color, (x, y, bar_width, bar_height))

        # Add particles for high-intensity bins
        if intensity > 0.1:
            for _ in range(2):  # Fewer particles for balance
                particle = Particle(
                    x + bar_width // 2,
                    y,
                    color,
                    np.random.uniform(5, 8),
                    (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
                )
                particles.append(particle)

    # --- Draw a dynamic and colorful waveform in the top-right corner ---

    # Waveform Area (Top-Right)
    waveform_width = WIDTH // 4
    waveform_height = HEIGHT // 4
    top_right_x = WIDTH - waveform_width - 10
    top_right_y = 10

    # Sample rate (adjust to your actual sample rate)
    sample_rate = 44100  # Assuming a sample rate of 44100 Hz

    # Prevent division by zero and NaN values in data_chunk
    if np.max(np.abs(data_chunk)) != 0:
        normalized_data = data_chunk / np.max(np.abs(data_chunk))
    else:
        normalized_data = np.zeros_like(data_chunk)  # Set to zero if data is all zeros

    # Smooth the data for a nicer appearance
    smoothed_data = smooth_waveform(normalized_data)

    # Get the dominant frequency from the audio data
    dominant_freq = get_dominant_frequency(smoothed_data, sample_rate)

    # Map the dominant frequency to a color
    base_color = map_frequency_to_color(dominant_freq)

    # Draw the waveform with color based on frequency
    scaled_data = smoothed_data * (waveform_height // 2)
    center_y = top_right_y + waveform_height // 2

    # Check for NaN or infinite values and replace with zero if necessary
    scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(1, len(smoothed_data)):
        x1 = top_right_x + int((i - 1) / len(smoothed_data) * waveform_width)
        y1 = center_y - int(scaled_data[i - 1])
        x2 = top_right_x + int(i / len(smoothed_data) * waveform_width)
        y2 = center_y - int(scaled_data[i])

        # Add smooth transition effect for color
        pygame.draw.line(screen, base_color, (x1, y1), (x2, y2), 2)

    # Draw a dynamic color-changing background for the waveform area
    pygame.draw.rect(screen, base_color, (top_right_x, top_right_y, waveform_width, waveform_height), 3)


def mic_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    return stream, audio

def file_stream(filename):
    if not filename.lower().endswith(".wav"):
        # Convert MP3 to WAV using pydub
        audio = AudioSegment.from_mp3(filename)  # Load MP3 file
        filename_wav = "temp.wav"  # Temporary WAV file
        audio.export(filename_wav, format="wav")  # Export as WAV
        filename = filename_wav  # Use the converted file
    
    wf = wave.open(filename, 'rb')  # Open the WAV file
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
    return stream, audio, wf

def run_visualizer(input_type, filename=None):
    if input_type == "mic":
        stream, audio = mic_stream()
        file_mode = False
    else:
        stream, audio, wf = file_stream(filename)
        file_mode = True

    running = True
    global particles
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if file_mode:
            data = wf.readframes(CHUNK)
            if len(data) == 0:
                running = False
                break
            stream.write(data)
        else:
            data = stream.read(CHUNK, exception_on_overflow=False)

        data = np.frombuffer(data, dtype=np.int16)
        visualize_fft(data)

        for particle in particles[:]:
            particle.move()
            particle.draw(screen)
            if particle.alpha <= 0:
                particles.remove(particle)

        pygame.display.flip()
        clock.tick(60)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    if file_mode:
        wf.close()
    pygame.quit()

def file_selection_window(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]

    font = pygame.font.SysFont("Arial", 20)
    buttons = []

    # Create buttons for each file in the folder
    for idx, file in enumerate(files):
        button = pygame.Rect(WIDTH // 4, HEIGHT // 4 + 50 * idx, WIDTH // 2, 40)
        buttons.append({"file": file, "rect": button})

    running = True
    while running:
        screen.fill((30, 30, 30))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if button["rect"].collidepoint(event.pos):
                        run_visualizer("file", os.path.join(folder_path, button["file"]))

        # Draw buttons for each file
        for button in buttons:
            pygame.draw.rect(screen, (100, 100, 200), button["rect"])
            label = font.render(button["file"], True, (255, 255, 255))
            screen.blit(label, (button["rect"].x + 20, button["rect"].y + 10))

        pygame.display.flip()
        clock.tick(30)

def main_menu():
    font = pygame.font.SysFont("Arial", 30)
    running = True
    buttons = [
        {"label": "Microphone Input", "rect": pygame.Rect(WIDTH//2 - 150, HEIGHT//2 - 80, 300, 50), "action": "mic"},
        {"label": "File Input", "rect": pygame.Rect(WIDTH//2 - 150, HEIGHT//2 + 10, 300, 50), "action": "file"},
        {"label": "Quit", "rect": pygame.Rect(WIDTH//2 - 150, HEIGHT//2 + 100, 300, 50), "action": "quit"}
    ]

    while running:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if button["rect"].collidepoint(event.pos):
                        if button["action"] == "mic":
                            run_visualizer("mic")
                        elif button["action"] == "file":
                            folder_path = "Audio Files"  # Specify your music folder here
                            file_selection_window(folder_path)
                        elif button["action"] == "quit":
                            running = False

        for button in buttons:
            color = (100, 100, 200) if button["rect"].collidepoint(pygame.mouse.get_pos()) else (70, 70, 150)
            pygame.draw.rect(screen, color, button["rect"])
            label = font.render(button["label"], True, (255, 255, 255))
            screen.blit(label, (button["rect"].x + 20, button["rect"].y + 10))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main_menu()
