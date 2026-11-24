# Cube Pickup with Custom Fine Tuned SmolVLA Model
## Setting up this project
1. Clone this repo, my dataset is not in the repo so you will need to create youre own if you plan on training, however my best trained checkpoint is included in the `/Best_Checkpoints` directory

#### Collecting Data
1. All manual data collection is done through a Xbox controller, simply plug any Xbox controller into your computer and it should automatically recognize (I specifically used an Xbox 1 controller).
2. Practice manipulating the arm by simply running the `conroller_demo.py` script, using `python controller_demo.py`.
3. To collect data run `python collect_data.py`, it will begin once you press `R` and reset every time you do so again. When you press `Q` it will save this current episode and exit the program.
4. Once data is collected run `python prepare_dataset.py` to convert the recordings to the LeRobot dataset format.
5. You are now ready to train

#### Training the model
* Run `python train_smolva.py` with parameters to train.
  * Parameters include `--dataset` (path to prepared dataset), `--output` (path to log directory, also where checkpoints are stored), `--checkpoint` (optional parameter if you want to train from a pre existing checkpoint), `--steps`, `--batch-size`

#### Testing
* Run `python policy_demo.py <path_to_checkpoint folder> <"prompt">` with parameters and it will run the simulator with the given model checkpoint.
  * Please make sure the prompt is enclosed with quotes
  * For example run `python .\policy_demo.py .\logs\checkpoints\010000 "Pick up the green cube and put it in the box"`
