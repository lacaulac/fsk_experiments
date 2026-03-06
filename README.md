# FSK Experimentations

`python .\main.py --help`

`python .\main.py --input .\main.py --output-audio test.wav`

`python .\main.py --input-audio .\test.wav --output recovered.py`

```
FSK Modem Demo

options:
  -h, --help            show this help message and exit
  -s SAMPLE_RATE, --sample-rate SAMPLE_RATE
                        Sample rate in Hz
  -a DATA_AMOUNT, --data-amount DATA_AMOUNT
                        Amount of symbols/bits to generate
  -r SYMBOL_RATE, --symbol-rate SYMBOL_RATE
                        Symbol rate in symbols per second
  --mark-freq MARK_FREQ
                        Mark frequency
  --space-freq SPACE_FREQ
                        Space frequency
  -n MIX_SNR_DB, --mix-snr-db MIX_SNR_DB
                        SNR in dB for mixing noise with signal
  --input INPUT         Path to input data file (optional)
  --output OUTPUT       Path to output data file (optional)
  --input-audio INPUT_AUDIO
                        Path to input audio file (optional)
  --output-audio OUTPUT_AUDIO
                        Path to output audio file (optional)
```