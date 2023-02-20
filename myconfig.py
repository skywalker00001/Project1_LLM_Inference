class Config():
    def __init__(self):
        self.root_dir = "/home/houyi/"
        self.working_dir=self.root_dir+"codes/Project1_LLM_Inference/"
        self.data_dir=self.root_dir+"data/crass/"
        self.date_version="1013_v1/"
        self.out_dir=self.root_dir+"outputs/crass_"+self.date_version
        self.data_name="crass_274_clean_v0.xlsx"
        self.data_path=self.data_dir+self.data_name
        self.off_loader_path=self.root_dir+"off_loader"

        self.device_ids=[0,1]
        #self.prompt_versions=[1,2,3,4,6,7]
        self.prompt_versions=[6]
        #self.prompt_versions=[1,4]
        self.seed=32
        self.inference_batch_size=8
        self.stop_sequence="Question"
        self.gpu_0_mem="23GiB"
        self.gpu_1_mem="38GiB"
        # self.gpu_0_mem="15GiB"
        # self.gpu_1_mem="25GiB"

        # set if it's debugging (log out all the infomation)
        self.is_debugging=False
        #self.use_less_data=True # need to change
        self.use_less_data=self.is_debugging
        self.save_group_num = 2 if self.use_less_data else 5  # need to change
        # self.save_group_num = 5  # need to change
        self.less_data_num= self.inference_batch_size * 1
        self.show_gpu_uti=self.is_debugging
        self.show_data_info=self.is_debugging
        self.log_text=self.is_debugging
        