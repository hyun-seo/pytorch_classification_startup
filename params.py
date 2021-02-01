import yaml
import os
import shutil


class Params:
    def __init__(self) -> None:
        self.exp_name = "debug_"
        self.model = "resnet18"
        self.dataset = "cifar10"
        self.prefetch = 12
        self.num_worker = 4
        self.seed = 2021

        # learning params
        self.batch_size = 128
        self.epoch = 300
        self.resume = "False"

        # opt params
        self.lr = 0.1
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.gamma = 0.1
        self.schedule = [150, 225]


    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

    def load(self, path):
        with open(path) as f:
            load_dict = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in load_dict.items():
            self.__setattr__(k, v)

    def __str__(self) -> str:
        _str = "==== params setting ====\n"
        for k, v in self.__dict__.items():
            _str += f"{k} : {v}\n"
        return _str

    def build(self) -> None:

        self.log_dir = os.path.join("./exps/logs", self.exp_name)
        self.save_dir = os.path.join(self.log_dir, "saved_model")
        self.tb_dir = os.path.join("./exps/tb", self.exp_name)
        if self.resume == "True":
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                os.makedirs(self.save_dir)
            if not os.path.exists(self.tb_dir):
                os.makedirs(self.tb_dir)
        else:
            if os.path.exists(self.log_dir):
                isdelete = input("delete exist exp dir (y/n): ")
                if isdelete == "y":
                    shutil.rmtree(self.log_dir)

                elif isdelete == "n":
                    raise FileExistsError
                else:
                    raise FileExistsError

            os.makedirs(self.log_dir)
            os.makedirs(self.save_dir)

            if os.path.exists(self.tb_dir):
                shutil.rmtree(self.tb_dir)
            os.makedirs(self.tb_dir)

        self.save(f"{self.log_dir}/params.yml")


if __name__ == "__main__":
    params = Params()
    params.save("./test.yml")
    params.load("./test.yml")
