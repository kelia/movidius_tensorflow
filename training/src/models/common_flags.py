from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('network_topology', 'simple', 'Network to train, "simple" or "residual-cnn"')

# Train parameters
flags.DEFINE_integer('img_width', 320, 'Target Image Width')
flags.DEFINE_integer('img_height', 240, 'Target Image Height')
flags.DEFINE_bool('sim_experiment', True, 'Flag to switch between sim/real world exp settings')
flags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("radius_normalization", 5.0, "Radius is normalized with this value")

flags.DEFINE_string('train_dir', "../training", 'Folder containing'
                     ' training experiments')
flags.DEFINE_string('val_dir', None, 'Folder containing'
                     ' validation experiments')
flags.DEFINE_string('checkpoint_dir', "../../graph_data/", "Directory name to"
                     "save checkpoints and logs.")

# Input Queues reading
flags.DEFINE_integer('num_threads', 8, 'Number of threads reading and '
                      '(optionally) preprocessing input files into queues')
flags.DEFINE_integer('capacity_queue', 100, 'Capacity of input queue. A high '
                      'number speeds up computation but requires more RAM')

# Log parameters
flags.DEFINE_integer("max_epochs", 100, "Maximum number of training epochs")

flags.DEFINE_bool('resume_train', False, 'Whether to restore a trained'
                   ' model for training')
flags.DEFINE_integer("summary_freq", 5, "Logging every log_freq iterations")
flags.DEFINE_integer("save_freq", 5, "Save the latest model every save_freq epochs")

# Testing parameters
flags.DEFINE_string('test_dir', "../testing", 'Folder containing'
                     ' testing experiments')
flags.DEFINE_string('output_dir', "./tests/test_0", 'Folder containing'
                     ' generated testing descriptors')
flags.DEFINE_string("ckpt_file", None, "Checkpoint file")
