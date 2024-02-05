import os
import random
from PIL import Image, ImageOps
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler

from utils.tokenizer import Tokenizer


def file_reader(directory, filename, sep=' '):
    examples = []
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line=line.strip("\n")
            if sep in line:
                # txt=[]
                # image_file = line.split(sep=sep, maxsplit=-1)[0] # Change for getting multiple label
                # txt.extend(line.split(sep=sep)[1:])
                image_file,txt= line.split(sep=sep, maxsplit=1)
                image_file = os.path.abspath(os.path.join(directory, image_file))
                txt = txt.strip()
                if os.path.isfile(image_file):
                    examples.append((txt, image_file))
    random.shuffle(examples)
    return examples  # list of tuple




class Image_OCR_Dataset(Dataset):
    def __init__(self, data_dir, filename, img_width, img_height, data_file_separator,max_len=20, chars=None):
        self.list_of_tuple = file_reader(data_dir, filename, sep=data_file_separator)  # list of tuple containing (label,image_file_path)
        print(len(self.list_of_tuple))


        self.img_trans = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
        ])

        self.max_len = max_len

        if chars is None:
            self.chars = list('一丁七万丈三上下不与丑世丘丙业东丝丢两丧个中临丸丹为主丽举久义之乌乐乒乔乖乘九乞习书买了予争事二于亏云互五井亚些亡交产亩亭亲人亿什仁仅仆仇今从他仗仙代以们件任仿企伏休众伙会伞伟传估似但位低住体何余作佣使侄侍侨侵便俎俗俘保信俩俭修俯俱倒倚借倦值倾假偏做健偶偷偿傅傍储儿允兄充兆光克免兔入全八公六共关兴兵其具典养兽内冈册再冒写军冠冤冬冲决冷冻净准凉凌减几凡凤凭凰凶出击凿刀分刊刑列刘刚创初利别刮到制刷剃前剖剥剧剩副加劣动助努劲劳勃勇勉勒勺勿匀包化北匙匪匹区医十千升午半华单卖南博卜占卡卧卫危即却卵卷卸厉压厌厕厚原厦厨去县又叉及友双反发叔叙口古另叫召叮可叱叶司叹叼吃吉同名后吐吓君吞否吧听吵吸吹吼呆呈呜呢周味命和咏咤咬咱咳品哄哈哨哪哭哲唇唐唤售唯唱啄商啊啦善喊喜喝嗨器嚎四回因团园困围国圈圣在地场圾均坏坐块坛坟坡坦垂垄型垒垦城培堂堡堵塔填境壁士壮声壳壶备复夕外够大天太夭央失头夸夺奂奇奏奔奖套奥女奴奶好如妄妖妙妥妨姐姑委姥姬姻姿娃娘娱娲婆婚婶嫂子孔孕存孟孤学孩宁它宅宇守安完宏宗官宙宝实审室宫害宴家容寄富寒对寻导寿射将尊小尖尘尝尤尸尼尽尾局屈屋展属屡屯山屿岁岂岔岗岭峡峰崖川巡左巨差己已巳巾币市布帅帆师希帘帝帮常干平年并幸幻幼广庄库应底庖店庙府废座庭康庸廊建开异弃弄式弓引弟张弯弱弹强归当形彩彰影役彼往待很律徒得御循德心必忆忌志忘忠忧快忽怎怒怖思性怨总恋恐恒恕恢恨恩恭息恰恶悄悉悟悠悦您悬悲情惊惕惜惧惩惭惯惰惹愉愎愚愤愧慌懈戈戏成我戒或战房所扁扇手扒打扔扛扣执扩扫扮扯扰扶批找承技抄抖抚护报披抱押担拉拌拐拒拔拘拜拢拣拥拦拳持指按挎挑挖挠挡挨挪挽捆捎捐捕损捡换捧捭据捷掀授掉掌掏掘探接掩描提插握揪援搁搅搜支改放政故效敌敏救敞敢散数文斑斗料斜斤斥斧斩断斯方施旁旅旋族无既日旦旧早时昌明昏易星映春昨是昼晌晒晓晚晦晨普晶智暑暖暧暴曲更曾替最月有朋服朗望朝期木末本术朱机杀权李杏村杜杞束条来杯杰极枕林果枝枣枯架枸某染柔柜查柯柱标栉栋栏栗校株核根格栽桂桃案桌桑桨桶梁梅梢梦梨械梳检棋棒棚椅植椒椟楚楼横欠次欢欣欧欲款歌止正此步死歼殄残殖段母每比毙毛毫毯民气水永汁求汉汗江池汤沈沉沙沟沦沧沫河沸治沾泉法泛泡波泥注泪泰泼泽洁洒洗津洲洽流浆浇测济浓浙浪浮浴海浸消涛涝润涯淋淑淘淡深混添清渉渔渠港渴游湾溉源滔满滥濡火灯灰灶灾灿炉炊炒炭炸点烂烈烘烦烧烫热焦然照煮燎燕爪爬爱父爷爸版牙牛牢物牲特牺犬犯犹狂狐狗狡独狮狸狼猛猜猫玉王玩环现玻珍珠球理琴瓴瓶甘甚甜生用田由甲申男画畅界留畜略疏疗疤疯疲痒登白百皆皇皱盈益盏盐监盒盖盗盛目盯直相盾眉看眠眨眯眼着睫矛知矩短石矿码砌砍破础确碧祝神祥票祸禁福禹离禽秀秃秆秋种秒秘租秧秩称移程税穴究穷空穿窈窍窑窕窜立竖站竞竟童竹竿笋笑笔笛符第笼等筋筐筒答箭簪粉粗粘粥粪精紧累纠约级纯纱纲纳纵纷纸纺线练组织绌经绑绒结绕绘给绝绞统绢继绩绪续绳绸绿编缘缚缨缪缸缺网罚罪羊美羡群羽翁翅翻老而耍耳耻聋职聚肃肉肌肚肝肠股肢肤肥肯育肺肿胁胃胆背胜胞胡胳胶胸脆脑脖脱脸脾腊腔腹自至致舌舍舒舞舟航舰舱船艇良艰色芒芦芬花芳芽苍苏苒若苦英苹茂范茄茧茶荆草荏荒荡荷荼莫莲获菊菌菜菠菲萄萌萍营落葛董葱葵薄薪虎虏虑虹虽虾蚀蚁蚂蚕蛇蛋蛙蜀蜃蜜蝉血行衔街衣补表衬衰衷袋袍袖装裙裤裹褚西要覆见观规览角解言计订让记讲许论讽设访评诉诊词译试诗诞询该语误诵请诸读课谅谈谊谋谢谣谷豆象豪貌贝负财责败货质贩贪贫贯贱贴贷贸费贺贿资赏赔赤走赴赶起超越足距路踏身躬车轧轨转轮载较辅辆辈辕辛辰辱边达迁过迎运还这进违迫述迷退适逃逆选逍透逐途逗通逝造逢逼遇遍道遗遥那邪邮邯郊郎郞部郸鄣配酒醉采释里野金釜针钉钓钞钟钢钥钧钳钻铁铃铅铜铲银销锁锈锐镇镜长门闪闭闯间闸闻阀阅阖防阳阻阿附际院险陪陵陶陷随隐隔隙隶难雀雄雨雪雷雾霆霸青非面革韬音韵页顶顷项顺须顽顿颈颜风飞食饭饮饰饿首香马驳驴驶驹驻驼驾骂骆验骗高鬼魅魍魑鳞鸟鸡鸩鸽鹰鹿麻黄黔齐齿龙龟ｄｆｊ')
        else:
            self.chars = list(chars)

        self.tokenizer = Tokenizer(self.chars)

        # self.first_run = True

    def __len__(self):
        return len(self.list_of_tuple)

    def __getitem__(self, idx):

        s = self.list_of_tuple[idx][0]
        d = Image.open(self.list_of_tuple[idx][1])

        label = torch.full((self.max_len + 2,), self.tokenizer.EOS_token, dtype=torch.long)

        ts = self.tokenizer.tokenize(s)
        #ts_shape = ts.shape
        label[:ts.shape[0]] = torch.tensor(ts)
        # label[:ts.shape] = torch.tensor(ts)

        return self.img_trans(d), label

if __name__ == "__main__":

    # img_width = 160
    # img_height = 60
    # max_len = 4
    #
    # nh = 512
    #
    # teacher_forcing_ratio = 0.5
    #
    # batch_size = 4
    #
    # lr = 3e-4
    # n_epoch = 10
    #
    # n_works = 1
    # save_checkpoint_every = 5
    #
    # data_dir = "C:/Users/Ankan/Desktop/pytorch_aocr/main_dir"
    # train_file = "train.txt"
    # test_file = "test.txt"
    #
    # max_len = 4
    #
    # ds_train = Image_OCR_Dataset(data_dir, train_file, img_width, img_height, 4, max_len)
    # print(ds_train.__len__())
    # tokenizer = ds_train.tokenizer
    #
    # train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=n_works)
    # print(len(train_loader.dataset))

    zeta=file_reader(r"/root/autodl-tmp/Attention-OCR-pytorch-main/58/train_resize", "train.txt", sep=',')
    print(zeta)