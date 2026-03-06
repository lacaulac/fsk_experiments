import numpy as np

import fastgoertzel as fg
import argparse
from time import time

def wave(amp: float, freq: float, phase: float, sample_number: np.ndarray, sample_rate: int) -> np.ndarray:
    """Generates a sine wave value for a given sample number.
    Args:
        amp (float): The amplitude of the sine wave.
        freq (float): The frequency of the sine wave in Hz.
        phase (float): The phase shift of the sine wave in radians.
        sample_number (np.ndarray): The indices of the samples for which to calculate the value.
        sample_rate (int): The number of samples per second.
        
        Returns:
        np.ndarray: The values of the sine wave at the given sample numbers."""
    return amp * np.sin((2 * np.pi * freq * (sample_number / sample_rate)) + phase)

def get_schedule(state_duration: int, data: np.ndarray, total_duration: int):
    mark_list = list()
    for current_step in range(total_duration):
        step_in_cycle = current_step % total_duration
        state_index = step_in_cycle // state_duration
        if data[state_index]:
            mark_list.append(1)
        else:
            mark_list.append(0)
    mark_nparray = np.array(mark_list)
    return mark_nparray, (1 - mark_nparray)

def generate_fsk_signal(sample_rate: int, mark_freq: int, space_freq: int, symbol_duration: int, total_duration: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Generates an FSK modulated signal based on the provided parameters and input data.

    Args:
        sample_rate (int): The number of samples per second in the generated signal.
        mark_freq (int): The frequency for the mark symbol.
        space_freq (int): The frequency for the space symbol.
        symbol_duration (int): The duration of each symbol in samples.
        total_duration (int): The total duration of the signal in samples.
        data (list[bool]): The input data bits to be modulated.

    Returns:
        tuple[np.ndarray, np.ndarray]: The time array and the modulated signal array.
    """
    t = np.arange(0, total_duration)
    print("[~] Generating base waveforms for mark and space frequencies...")
    y1 = wave(1, mark_freq, 0, t, sample_rate)
    y2 = wave(.8, space_freq, 0, t, sample_rate)
    print("[+] Base waveforms generated!")
    print("[~] Generating mark and space schedules based on input data...")
    mark_schedule, space_schedule = get_schedule(symbol_duration, data, total_duration)
    print("[+] Schedules generated!")
    print("[~] Combining waveforms with schedules to create final FSK signal...")
    signal_output = (y1 * mark_schedule) + (y2 * space_schedule)
    print("[+] Final FSK signal created!")
    return t,signal_output

def load_bits_from_file(file_path):
    with open(file_path, 'rb') as f:
        file_data = np.frombuffer(f.read(), dtype=np.uint8)
        file_data = np.unpackbits(file_data)[:].astype(bool)
    return file_data

def generate_noise_for_signal(desired_snr, signal):
    noise = np.random.normal(0, 0.5, len(signal))
    signal_power = np.mean(np.array(signal)**2)
    noise_power = signal_power / (10**(desired_snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = np.array(signal) + noise
    #Normalize the noisy signal to be between -1 and 1
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
    return noisy_signal

def goertzel_analysis(sample_rate, mark_freq, block_size, t, input_signal):
    analysis_result: list[float] = list()
    for i in range(0, len(t), block_size):
        block = np.array(input_signal[i:i+block_size])
        res = fg.goertzel(block, mark_freq/sample_rate)
        analysis_result.append(res[0])
    return analysis_result

def postprocess_goertzel(goertzel_result):
    threshold = np.percentile(goertzel_result, 97)
    print(f"\t[+] 97th Percentile of Goertzel Result: {threshold:.2f}")
    goertzel_result = goertzel_result/threshold # type: ignore

    #"Normalise" the result to binary values
    detected = [1 if r > 0.5 else 0 for r in goertzel_result]
    return detected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSK Modem Demo")
    parser.add_argument('-s', '--sample-rate', type=int, default=44100, help='Sample rate in Hz')
    parser.add_argument('-a', '--data-amount', type=int, default=10000, help='Amount of symbols/bits to generate')
    parser.add_argument('-r', '--symbol-rate', type=int, default=300, help='Symbol rate in symbols per second')
    parser.add_argument('--mark-freq', type=int, default=2200, help='Mark frequency')
    parser.add_argument('--space-freq', type=int, default=1200, help='Space frequency')
    parser.add_argument('-n', '--mix-snr-db', type=int, default=0, help='SNR in dB for mixing noise with signal')
    parser.add_argument('--input', type=str, help='Path to input data file (optional)')
    parser.add_argument('--output', type=str, help='Path to output data file (optional)')
    parser.add_argument('--input-audio', type=str, help='Path to input audio file (optional)')
    parser.add_argument('--output-audio', type=str, help='Path to output audio file (optional)')
    args = parser.parse_args()
    DATA_AMOUNT = args.data_amount
    SYMBOL_RATE = args.symbol_rate
    MAIN_FREQ = args.mark_freq
    OTHER_FREQ = args.space_freq
    MIX_SNR_DB = args.mix_snr_db
    INPUT_FILE_PATH = args.input
    OUTPUT_FILE_PATH = args.output
    OUTPUT_AUDIO_PATH = args.output_audio
    INPUT_AUDIO_PATH = args.input_audio
    start_time = time()
    if not INPUT_AUDIO_PATH:
        SAMPLE_RATE = args.sample_rate
        SYMBOL_DURATION = SAMPLE_RATE // SYMBOL_RATE
        TOTAL_DURATION = SYMBOL_DURATION * DATA_AMOUNT
        if INPUT_FILE_PATH:
            print(f"[~] Loading input data from {INPUT_FILE_PATH}...")
            input_data = load_bits_from_file(INPUT_FILE_PATH)
            DATA_AMOUNT = len(input_data)
            TOTAL_DURATION = SYMBOL_DURATION * DATA_AMOUNT
            print("[+] Input data loaded!")

        print(f"Total duration in seconds: {TOTAL_DURATION / SAMPLE_RATE:.2f}")
        print(f"Total number of samples: {TOTAL_DURATION}")
        print(f"Symbols per second: {SYMBOL_RATE}")
        print(f"Samples per second: {SAMPLE_RATE}")
        print(f"Samples per symbol: {SYMBOL_DURATION}")
        print(f"Symbol amount: {DATA_AMOUNT}")



        if not INPUT_FILE_PATH:
            print(f"[~] Generating {DATA_AMOUNT} random symbols...")
            input_data = np.random.choice([True, False], size=DATA_AMOUNT)
            print("[+] Data generation complete!")

        print("[~] Generating FSK modulated signal")
        x, y = generate_fsk_signal(SAMPLE_RATE, MAIN_FREQ, OTHER_FREQ, SYMBOL_DURATION, TOTAL_DURATION, input_data)
        print("[+] FSK modulated signal generated")

        print(f"[~] Generating noise and mixing with signal at SNR of {MIX_SNR_DB} dB")
        # Generate Gaussian noise for the duration of the signal
        noisy_signal = generate_noise_for_signal(MIX_SNR_DB, y)

        y = noisy_signal

        print("[+] Noise generated and mixed with signal")

        if OUTPUT_AUDIO_PATH:
            print(f"[~] Saving generated signal to {OUTPUT_AUDIO_PATH}...")
            from scipy.io import wavfile
            wavfile.write(OUTPUT_AUDIO_PATH, SAMPLE_RATE, (noisy_signal * 32767).astype(np.int16))
            print("[+] Signal saved!")
    else:
        if not OUTPUT_FILE_PATH:
            OUTPUT_FILE_PATH = "decoded_output.bin"
        print(f"[~] Loading audio signal from {INPUT_AUDIO_PATH}...")
        from scipy.io import wavfile
        SAMPLE_RATE, y = wavfile.read(INPUT_AUDIO_PATH)
        y = y / 32767  # Normalize to -1 to 1
        x = np.arange(len(y))
        print("[+] Audio signal loaded!")
        
        SYMBOL_DURATION = SAMPLE_RATE // SYMBOL_RATE
        TOTAL_DURATION = SYMBOL_DURATION * DATA_AMOUNT
        
        print(f"Total duration in seconds: {TOTAL_DURATION / SAMPLE_RATE:.2f}")
        print(f"Total number of samples: {TOTAL_DURATION}")
        print(f"Symbols per second: {SYMBOL_RATE}")
        print(f"Samples per second: {SAMPLE_RATE}")
        print(f"Samples per symbol: {SYMBOL_DURATION}")
        print(f"Symbol amount: {DATA_AMOUNT}")

    print("[~] Analyzing signal with Goertzel algorithm...")
    block_size = SYMBOL_DURATION
    result = goertzel_analysis(SAMPLE_RATE, MAIN_FREQ, block_size, x, y)
    print("[+] Goertzel analysis complete!")

    print("[~] Post-processing Goertzel results...")
    detected = postprocess_goertzel(result) 
    print("[+] Post-processing complete!")

    print("[~] Computing decoded symbols from detected blocks...")
    # Average each detected block to get one value per symbol
    DETECTS_PER_STATE = SYMBOL_DURATION // block_size
    decoded = [True if sum(detected[i:i+DETECTS_PER_STATE]) > (DETECTS_PER_STATE // 2) else False for i in range(0, len(detected), DETECTS_PER_STATE)]
    print("[+] Fully decoded!")
    if not INPUT_AUDIO_PATH:
        # Compute the bit error rate
        bit_errors = sum(d != o for d, o in zip(decoded, input_data))
        bit_error_rate = bit_errors / len(input_data)
        print(f"Bit error rate: {bit_error_rate:.2%} ({bit_errors} errors out of {len(input_data)} bits)")
    end_time = time()
    if OUTPUT_FILE_PATH:
        print(f"[~] Saving decoded data to {OUTPUT_FILE_PATH}...")
        with open(OUTPUT_FILE_PATH, 'wb') as f:
            byte_data = np.packbits(decoded).tobytes()
            f.write(byte_data)
        print("[+] Decoded data saved!")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
