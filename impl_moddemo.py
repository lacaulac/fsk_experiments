import numpy as np
import pandas as pd

import fastgoertzel as fg
import argparse
from time import time

def wave(amp, freq, phase, sample_number, sample_rate):
    return amp * np.sin((2 * np.pi * freq * (sample_number / sample_rate)) + phase)

def scheduler(state_duration: int, data, current_step):
    total_duration = state_duration * len(data)
    step_in_cycle = current_step % total_duration
    state_index = step_in_cycle // state_duration
    return data[state_index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSK Modem Demo")
    parser.add_argument('-s', '--sample-rate', type=int, default=44100, help='Sample rate in Hz')
    parser.add_argument('-a', '--data-amount', type=int, default=10000, help='Amount of symbols/bits to generate')
    parser.add_argument('-r', '--symbol-rate', type=int, default=300, help='Symbol rate in symbols per second')
    parser.add_argument('--mark-freq', type=int, default=2200, help='Mark frequency')
    parser.add_argument('--space-freq', type=int, default=1200, help='Space frequency')
    parser.add_argument('-n', '--mix-snr-db', type=int, default=0, help='SNR in dB for mixing noise with signal')
    args = parser.parse_args()
    SAMPLE_RATE = args.sample_rate
    DATA_AMOUNT = args.data_amount
    SYMBOL_RATE = args.symbol_rate
    MAIN_FREQ = args.mark_freq
    OTHER_FREQ = args.space_freq
    MIX_SNR_DB = args.mix_snr_db
    SYMBOL_DURATION = SAMPLE_RATE // SYMBOL_RATE
    TOTAL_DURATION = SYMBOL_DURATION * DATA_AMOUNT

    print(f"Total duration in seconds: {TOTAL_DURATION / SAMPLE_RATE:.2f}")
    print(f"Total number of samples: {TOTAL_DURATION}")
    print(f"Symbols per second: {SYMBOL_RATE}")
    print(f"Samples per second: {SAMPLE_RATE}")
    print(f"Samples per symbol: {SYMBOL_DURATION}")
    print(f"Symbol amount: {DATA_AMOUNT}")
    
    start_time = time()

    print(f"[~] Generating {DATA_AMOUNT} random symbols...")
    input_data = np.random.choice([True, False], size=DATA_AMOUNT)
    print("[+] Data generation complete!")

    print("[~] Generating FSK modulated signal")
    x = np.arange(0, TOTAL_DURATION)
    y1 = wave(1, MAIN_FREQ, 0, x, SAMPLE_RATE)
    y2 = wave(.8, OTHER_FREQ, 0, x, SAMPLE_RATE)
    y = [(y1[i] if scheduler(SYMBOL_DURATION, input_data, i) else y2[i]) for i in range(len(x))]
    print("[+] FSK modulated signal generated")

    print(f"[~] Generating noise and mixing with signal at SNR of {MIX_SNR_DB} dB")
    # Generate Gaussian noise for the duration of the signal
    noise = np.random.normal(0, 0.5, len(y))
    signal_power = np.mean(np.array(y)**2)
    noise_power = signal_power / (10**(MIX_SNR_DB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    noisy_signal = np.array(y) + noise
    #Normalize the noisy signal to be between -1 and 1
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
    noisy_y = noisy_signal.tolist()

    y = noisy_y

    print("[+] Noise generated and mixed with signal")

    print("[~] Analyzing signal with Goertzel algorithm...")
    BLOCK_SIZE = SYMBOL_DURATION//1
    result: list[float] = list()
    for i in range(0, len(x), BLOCK_SIZE):
        block = np.array(y[i:i+BLOCK_SIZE])
        res = fg.goertzel(block, MAIN_FREQ/SAMPLE_RATE)
        result.append(res[0])
    print("[+] Goertzel analysis complete!")

    print("[~] Post-processing Goertzel results...")
    # #Normalize the result to be between 0 and 1
    # result = result / np.max(result)
    # Get the top 97%
    threshold = np.percentile(result, 97)
    print(f"\t[+] 97th Percentile of Goertzel Result: {threshold:.2f}")
    result = result/threshold # type: ignore

    #"Normalise" the result to binary values
    detected = [1 if r > 0.5 else 0 for r in result] 
    print("[+] Post-processing complete!")

    print("[~] Computing decoded symbols from detected blocks...")
    # Average each detected block to get one value per symbol
    DETECTS_PER_STATE = len(result) // len(input_data)
    decoded = [True if sum(detected[i:i+DETECTS_PER_STATE]) > (DETECTS_PER_STATE // 2) else False for i in range(0, len(detected), DETECTS_PER_STATE)]
    print("[+] Fully decoded!")
    # Compute the bit error rate
    bit_errors = sum(d != o for d, o in zip(decoded, input_data))
    bit_error_rate = bit_errors / len(input_data)
    print(f"Bit error rate: {bit_error_rate:.2%} ({bit_errors} errors out of {len(input_data)} bits)")
    end_time = time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")