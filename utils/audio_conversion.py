# mel.py function is adopted and modified based on top of audio-diffusion repository 
# reference: https://github.com/teticio/audio-diffusion/blob/main/audiodiffusion/mel.py
import argparse
import io
import logging
import os
import re
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
# from diffusers.pipelines.audio_diffusion import Mel  
from mel import Mel
from tqdm.auto import tqdm
import shutil
from scipy.io.wavfile import write  # For saving audio
from PIL import Image



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("audio_to_images")


def process_to_disk(input_dir, output_dir, mel):
    examples = []
    # get the list of all files directly in input_dir parameter, since traverse_directies handles recursive call 
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # don't display progress bar if not file inside parent directories 
    if len(files) == 0:
        return

    with tqdm(total=len(files), unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
        for file in files:
            if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE):
                audio_file = os.path.join(input_dir, file)
                try:
                    mel.load_audio(audio_file)
                except:
                    continue

                for slice in range(mel.get_number_of_slices()):
                    image = mel.audio_slice_to_image(slice)
                    if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                        continue
                    with io.BytesIO() as output:
                        image.save(output, format="PNG")
                        bytes = output.getvalue()

                    examples.append(
                        {
                            "image": {"bytes": bytes},
                            "audio_file": audio_file,
                            "slice": slice,
                        }
                    )
            t.update(1)

    if len(examples) == 0:
        return

    ds = Dataset.from_pandas(
        pd.DataFrame(examples),
        features=Features(
            {
                "image": Image(),
                "audio_file": Value(dtype="string"),
                "slice": Value(dtype="int16"),
            }
        ),
    )
    dsd = DatasetDict({"train": ds})
    # you can save to disk for next time use
    # ----------------------------
    dsd.save_to_disk(output_dir)
    # ----------------------------
    # if args.push_to_hub:
    #     dsd.push_to_hub(args.push_to_hub)




# save image file
def process_to_directory(input_dir, output_dir, mel):
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # don't display progress bar if not file inside parent directories 
    if len(files) == 0:
        return

    for file in tqdm(files, desc=f"Processing {input_dir}", unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->='):
        if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE):
            audio_file = os.path.join(input_dir, file)
            try:
                mel.load_audio(audio_file)
            except:
                continue

            # print("slices: ", mel.get_number_of_slices())
            for slice in range(mel.get_number_of_slices()):
                image = mel.audio_slice_to_image(slice)
                if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                    continue

                # image format conversion, temporarily line for converting grayscale to RGB
                # default is GrayScale
                # image = image.convert("RGB")  # remove later when grayscale model is done

                # adding slices extension here, because when audio file is longer, you need to slice it 
                # and you need to know how many slices are there, and what part is what slice
                image_filename = os.path.splitext(os.path.basename(audio_file))[0] + f"_slice_{slice}.png"
                # image_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".png"
                save_path = os.path.join(output_dir, image_filename)
                image.save(save_path, "PNG")
    return



def process_to_audio(input_dir, output_dir, mel):
    # print(mel.get_sample_rate())
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # don't display progress bar if not file inside parent directories 
    if len(files) == 0:
        return

    for file in tqdm(files, desc=f"Processing {input_dir}", unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->='):
        if file.endswith(".png"):
            image_file = os.path.join(input_dir, file)
            
            # Load the image
            image = Image.open(image_file).convert("L")
            
            # Convert the image to audio using the Mel class method
            audio = mel.image_to_audio(image)
            
            # Generate the output audio file name
            audio_filename = os.path.splitext(os.path.basename(image_file))[0] + ".wav"
            save_path = os.path.join(output_dir, audio_filename)
            
            # Save the audio to disk
            write(save_path, mel.get_sample_rate(), audio)

       

# split the mixed file based on their class(assuming the first filename is the class name)
# e.g. dog_001.png --> dog/
def separate_images_by_class(src_dir):
    for filename in os.listdir(src_dir):
        if filename.endswith(".png"):
            class_name = filename.split("_")[0]
            dest_dir = os.path.join(src_dir, class_name)
            
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            shutil.move(src_path, dest_path)



# walk through all the files for directories, and subdirectories
def traverse_directories(root_dir, output_dir, mel, wav=False, save_to_disk=False):

    # process root directory where audio/mel files located, bulid the path
    current_output_dir = os.path.join(output_dir, os.path.basename(root_dir))
    # create one if the given one doesn't exist
    os.makedirs(current_output_dir, exist_ok=True)

    # check conditions, for single directory (root)
    if wav:
        process_to_audio(root_dir, current_output_dir, mel)
    elif save_to_disk:
        process_to_disk(root_dir, current_output_dir, mel)
    else:
        process_to_directory(root_dir, current_output_dir, mel)
        separate_images_by_class(current_output_dir)                # just for speech command, remove it if needed

    # after root has been check, if there exists some nested directory
    # each of them will become new root for new checking round 
    for root, _, _ in os.walk(root_dir):
        if root == root_dir:
            continue 
        # build relative path
        rel_path = os.path.relpath(root, root_dir)
        current_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)
        if wav: 
            process_to_audio(root, current_output_dir, mel)
        elif save_to_disk:
            process_to_disk(root, current_output_dir, mel)
        else:
            process_to_directory(root, current_output_dir, mel)
            # design just for speech command dataset, remove it if don't want to separate
            separate_images_by_class(current_output_dir)



def main(args):

    if args.wav:
        print("###########################################")
        print("#                                         #")
        print("#  Mel-Spectrogram to Audio Conversion..  #")
        print("#                                         #")
        print("###########################################")

    else:
        print("###########################################")
        print("#                                         #")
        print("#  Audio to Mel-Spectrogram Conversion..  #")
        print("#                                         #")
        print("###########################################")


    # initalized Mel object
    mel = Mel(
        x_res=args.resolution[0],
        y_res=args.resolution[1],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

    # if single file passed into for conversion 
    if args.file:
        # create the directory if not existed
        os.makedirs(args.output_dir, exist_ok=True)

        audio_file = args.file
        if re.search("\.(mp3|wav|m4a)$", audio_file, re.IGNORECASE):
            try:
                mel.load_audio(audio_file)
            except Exception as e:
                print(f"Failed to load audio: {e}")
                return

            num_slices = mel.get_number_of_slices()
            with tqdm(total=num_slices, unit='slice', bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
                for slice in range(num_slices):
                    image = mel.audio_slice_to_image(slice)
                    image_filename = os.path.splitext(os.path.basename(audio_file))[0] + f"_slice_{slice}.png"
                    save_path = os.path.join(args.output_dir, image_filename)
                    image.save(save_path, "PNG")
                    t.update(1)  # Update the tqdm progress bar

        else:
            print("Invalid file type. Only .mp3, .wav, and .m4a are supported\nIf passing directory use --input_dir")

        return 

    # traverse directories 
    traverse_directories(args.input_dir, args.output_dir, mel, wav=args.wav, save_to_disk=args.save_to_disk)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--file", type=str, default=None, help="Single audio file for processing.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing audio files.")
    parser.add_argument("--wav", action='store_true', help="indicating mel-spectrogram back to .wav file")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--save_to_disk", type=bool, default=False)
    parser.add_argument(
        "--resolution",
        type=str,
        default="256",
        help="Either square resolution or width,height.",
    )
    parser.add_argument("--hop_length", type=int, default=512)
    # parser.add_argument("--hop_length", type=int, default=1024)
    # parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    # parser.add_argument("--push_to_hub", type=str, default=None)
    
    args = parser.parse_args()
    
    # input directory is not required, but either one is required, --input_dir or --file
    if args.input_dir is None and args.file is None:
        raise ValueError("You must specify either an input directory or an input file for the audio files.")
    
    try:
        args.resolution = (int(args.resolution), int(args.resolution))
    except ValueError:
        try:
            args.resolution = tuple(int(x) for x in args.resolution.split(","))
            if len(args.resolution) != 2:
                raise ValueError
        except ValueError:
            raise ValueError("Resolution must be a tuple of two integers or a single integer.")
    
    assert isinstance(args.resolution, tuple)
    
    main(args)
