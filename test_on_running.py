import os
import logging
import numpy as np
import keras
import pandas as pd
import tensorflow as tf
import time
from model_structure import create_s2p_model, load_model, create_mobilenet_model
from data_feeder import TestSlidingWindowGenerator
from appliance_data import appliance_data, mains_data
import matplotlib.pyplot as plt
import nilm_metric as nm
import threading

mae_all = []
saex_all = []
saey_all = []


class Tester():
    def __init__(self, appliance, algorithm, crop, batch_size, network_type,
                 test_directory, saved_model_dir, log_file_dir,
                 input_window_length, plot, datadir):
        self.__appliance = appliance
        self.__algorithm = algorithm
        self.__network_type = network_type

        self.__crop = crop  # #crop#cropNone
        self.__batch_size = batch_size
        self._input_window_length = input_window_length
        self.__window_size = self._input_window_length
        self.__window_offset = int(self.__window_size // 2)
        self.__number_of_windows = batch_size  # 1000

        self.__test_directory = test_directory
        self.__saved_model_dir = saved_model_dir
        self.__plot = plot
        self.__log_file = log_file_dir
        self.__datadir = datadir
        self.max_number_of_windows = 60000

        logging.basicConfig(filename=self.__log_file, level=logging.INFO)

    def test_model(self):
        tflite_test = True
        '''config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)'''
        logging.info("---------" + self.__saved_model_dir + "----------")
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        test_input, test_target = self.load_dataset(self.__test_directory)
        self.__test_input = test_input
        self.__test_target = test_target
        # --------------------------------Keras Model--------------------------------------------#
        model = create_mobilenet_model(self._input_window_length)  # _s2p
        model = load_model(model, self.__network_type, self.__algorithm,
                           self.__appliance, self.__saved_model_dir)
        model.summary()
        test_generator = TestSlidingWindowGenerator(number_of_windows=self.__number_of_windows, inputs=test_input,
                                                    targets=test_target, offset=self.__window_offset,
                                                    qo=self._input_window_length % 2)

        # Calculate the optimum steps per epoch.
        import math
        steps_per_test_epoch = np.round(math.ceil(test_generator.total_size / self.__batch_size), decimals=0)

        # Test the model.
        start_time = time.time()

        testing_history = model.predict(x=test_generator.load_dataset(), steps=steps_per_test_epoch, verbose=1)  #

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.allow_custom_ops = True
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float32]
        tflite_model = converter.convert()

        open("./saved_models/" + self.__datadir + "_s/" + self.__appliance + "_model.tflite", "wb").write(tflite_model)

        end_time = time.time()
        test_time = end_time - start_time
        print("runtime: ", test_time / self.__batch_size, "(sample per sec)")  # / self.__crop

        self.plot_results(testing_history, test_input, test_target)
        pre_data = []
        tar_data = []
        if tflite_test:
            thread = []
            nthread = 8
            maesum = 0
            saexsum = 0
            saeysum = 0
            if self.max_number_of_windows * nthread > test_target.size:
                self.max_number_of_windows = int(test_target.size / nthread - 5)
            for i in range(nthread):
                thread.append(
                    tfThread(threadID=i, inputs=self.__test_input, outputs=self.__test_target, datadir=self.__datadir,
                             appliance=self.__appliance, max_number_of_windows=self.max_number_of_windows,
                             pre_data=pre_data, tar_data=tar_data))
                thread[i].setDaemon(True)
                thread[i].start()
                mae_all.append(0)
                saey_all.append(0)
                saex_all.append(0)
                pre_data.append([])
                tar_data.append([])
            for i in range(nthread):
                thread[i].join()
            for i in range(nthread):
                maesum += mae_all[i]
                saexsum += saex_all[i]
                saeysum += saey_all[i]
            saesum = abs(saexsum - saeysum) / saexsum
            print("--MAE=" + str(maesum / nthread / self.max_number_of_windows) + "; SAE=" + str(
                saesum / nthread) + ";--\n")
            logging.info("--MAE=" + str(maesum / nthread / self.max_number_of_windows) + "; SAE=" + str(
                saesum / nthread) + ";--\n")
            pre_data = np.array(pre_data)
            tar_data = np.array(tar_data)
            save_name = "saved_models/" + self.__datadir + '_s/' + self.__appliance
            np.save(save_name + '_pred.npy', pre_data.flatten())
            np.save(save_name + '_gt.npy', tar_data.flatten())
            np.save(save_name + '_mains.npy', np.array(test_input).flatten())

    def load_dataset(self, directory):

        data_frame = pd.read_csv(directory, nrows=self.__crop, skiprows=self.__crop * 0, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        test_output_seq = np.round(np.array(data_frame.iloc[:, 1], float), 6)  # seq2seq
        test_target = np.round(np.array(data_frame.iloc[self.__window_offset: -self.__window_offset, 1], float),
                               6)  # seq2point
        del data_frame
        return test_input, test_target  # test_output_seq

    def log_results(self, model, test_time, evaluation_metrics):

        inference_log = "Inference Time: " + str(test_time)
        logging.info(inference_log)

        metric_string = "MSE: ", str(evaluation_metrics[0]), " MAE: ", str(evaluation_metrics[3])
        logging.info(metric_string)

        self.count_pruned_weights(model)

    def count_pruned_weights(self, model):

        num_total_zeros = 0
        num_dense_zeros = 0
        num_dense_weights = 0
        num_conv_zeros = 0
        num_conv_weights = 0
        for layer in model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                layer_weights = layer.get_weights()[0].flatten()

                if "conv" in layer.name:
                    num_conv_weights += np.size(layer_weights)
                    num_conv_zeros += np.count_nonzero(layer_weights == 0)

                    num_total_zeros += np.size(layer_weights)
                else:
                    num_dense_weights += np.size(layer_weights)
                    num_dense_zeros += np.count_nonzero(layer_weights == 0)

        conv_zeros_string = "CONV. ZEROS: " + str(num_conv_zeros)
        conv_weights_string = "CONV. WEIGHTS: " + str(num_conv_weights)
        conv_sparsity_ratio = "CONV. RATIO: " + str(num_conv_zeros / num_conv_weights)

        dense_weights_string = "DENSE WEIGHTS: " + str(num_dense_weights)
        dense_zeros_string = "DENSE ZEROS: " + str(num_dense_zeros)
        dense_sparsity_ratio = "DENSE RATIO: " + str(num_dense_zeros / num_dense_weights)

        total_zeros_string = "TOTAL ZEROS: " + str(num_total_zeros)
        total_weights_string = "TOTAL WEIGHTS: " + str(model.count_params())
        total_sparsity_ratio = "TOTAL RATIO: " + str(num_total_zeros / model.count_params())

        print("LOGGING PATH: ", self.__log_file)

        logging.info(conv_zeros_string)
        logging.info(conv_weights_string)
        logging.info(conv_sparsity_ratio)
        logging.info("")
        logging.info(dense_zeros_string)
        logging.info(dense_weights_string)
        logging.info(dense_sparsity_ratio)
        logging.info("")
        logging.info(total_zeros_string)
        logging.info(total_weights_string)
        logging.info(total_sparsity_ratio)

    def plot_results(self, testing_history, test_input, test_target):

        testing_history = (
                    (testing_history * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance][
                "mean"])
        test_target = (
                    (test_target * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance]["mean"])
        test_agg = (test_input.flatten() * mains_data["std"]) + mains_data["mean"]
        test_agg = test_agg[:testing_history.size]

        # Can't have negative energy readings - set any results below 0 to 0.
        test_target[test_target < 0] = 0
        testing_history[testing_history < 0] = 0
        test_input[test_input < 0] = 0

        sample_second = 8
        metric_string = "SAE: {:}\n".format(nm.get_sae(test_target.flatten(), testing_history.flatten(), sample_second)) \
                        + "SAEsita: {:}\n".format(
            nm.get_saesita(test_target.flatten(), testing_history.flatten(), sample_second)) \
                        + 'NDE:{0}\n'.format(nm.get_nde(test_target.flatten(), testing_history.flatten())) \
                        + '\n\n-----MAE: {:}\n    -std: {:}\n    -min: {:}\n    -max: {:}\n    -q1: {:}\n    -median: {:}\n    -q2: {:}\n'.format(
            *nm.get_abs_error(test_target.flatten(), testing_history.flatten())) \
                        + 'Energy per Day: {:}\n'.format(
            nm.get_Epd(test_target.flatten(), testing_history.flatten(), sample_second))
        logging.info(metric_string)
        print(metric_string)
        logging.info('\n#####' + time.asctime(time.localtime(time.time())) + '#####\n')
        '''save_name = "saved_models/"+ self.__datadir + '_s/' + self.__appliance
        np.save(save_name + '_pred.npy', testing_history.flatten())
        np.save(save_name + '_gt.npy', test_target.flatten())
        np.save(save_name + '_mains.npy', test_agg.flatten())'''
        # Plot testing outcomes against ground truth.
        plt.figure(1)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()
        if self.__plot:
            plt.show()


class tfThread(threading.Thread):
    def __init__(self, threadID, inputs, outputs, datadir, appliance, max_number_of_windows, pre_data, tar_data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.inputs = inputs
        self.outputs = outputs
        self.__window_offset = 299
        self.__datadir = datadir
        self.__appliance = appliance
        self.max_number_of_windows = max_number_of_windows
        self.pre_data = pre_data
        self.tar_data = tar_data

    def run(self):
        # --------------------------------Keras Model--------------------------------------------#
        # self.__crop
        interpreter = tf.lite.Interpreter(
            model_path="saved_models/" + self.__datadir + "_s/" + self.__appliance + "_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        ts = interpreter.get_tensor_details()
        input_index = input_details[0]['index']
        saex = 0
        saey = 0
        sae = 0
        mae = 0
        pre = []
        tar = []
        test_input = self.inputs
        test_target = self.outputs
        skipr = self.max_number_of_windows * self.threadID
        for index in range(skipr, self.max_number_of_windows + skipr, 1):  # + self.__window_offset
            input_data = np.array(test_input[index: index + 2 * self.__window_offset + 1]).astype(np.float32)
            target_data = np.array(test_target[index + self.__window_offset]).reshape(-1, 1)  # + self.__offset
            input_data = input_data.reshape(1, 599)
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = ((output_data * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance][
                "mean"])
            output_data[output_data < 0] = 0
            target_data = ((target_data * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance][
                "mean"])
            pre.append(output_data)
            tar.append(target_data)
            saex += target_data
            saey += output_data
            mae += abs(target_data - output_data)
        # pre = np.array(pre).squeeze()
        # tar = np.array(tar).squeeze()
        self.pre_data[self.threadID] = pre
        self.tar_data[self.threadID] = tar
        print("MAE=", mae, "; SAE=", sae, ";\n")
        mae_all[self.threadID] = mae
        saex_all[self.threadID] = saex
        saey_all[self.threadID] = saey


class DoubleSourceProvider3(object):

    def __init__(self, nofWindows, offset):
        self.nofWindows = nofWindows
        self.offset = offset

    def feed(self, inputs):
        inputs = inputs.flatten()
        max_nofw = inputs.size - 2 * self.offset

        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])

            yield inp
