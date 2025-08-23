# conda activate wan
export CUDA_VISIBLE_DEVICES=3
csv_num=1
# prompt_num=800
csv_file="./t2v_pair_prompts_$csv_num.csv"
# json_root='/storage/pmj/data_paul/Wan2.1/wan_caption'
wan_output_root='./output'
first_frame_root='./first_frame'
# cd ./Wan2.1
# python prompt_generate.py --csv_file "$csv_file" --prompt_num $prompt_num
# python caption_gallery.py --csv_file "$csv_file" --rootdir "$json_root"


# cd ./Wan2.2
python prompt_guide_generate.py --file_name "$csv_file" --output_root "$wan_output_root"


wan_output_folder=$(ls -dt $wan_output_root/output_${csv_num}_* | head -n 1)
python convert.py --input_dir $wan_output_folder --first_frame_root $first_frame_root