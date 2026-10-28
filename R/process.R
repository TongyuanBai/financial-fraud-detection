
#载入数据
db <- read.csv('E:/essay/data/data_clean.csv', encoding = "UTF-8")

mytable <- with(db,table(db$FLAG))
df2_1 <- as.data.frame(mytable)
library(ggplot2)
ggplot(data=df2_1, mapping=aes(x=Var1,y=Freq,fill=Var1))+
  geom_bar(stat="identity") + theme_bw() + 
  labs(x="class", y="Frequency", fill = "Whether company have financial fraud or not") +
  theme(panel.grid = element_blank(),legend.position = "bottom") +
  geom_text(aes(label=Freq), vjust=1.6, color="black", size=3.5)

mytable2 <- with(db,table(db$category))
df2_2 <- as.data.frame(mytable2)
ggplot(df2_2,aes(x=Var1,y=Freq,fill=Var1))+
  geom_bar(stat='identity',width=0.5)+
  labs(x="公司类型", y="频数", fill = "公司类型")+
  geom_hline(yintercept=0)+geom_text(aes(label=Freq), hjust=-0.25, color="black", size=3.5)+
  coord_flip()+
  theme_bw()+
  theme(panel.grid.major.x=element_blank(),panel.grid.minor.x=element_blank(), #去除网格
        panel.grid.minor.y=element_blank(),panel.grid.major.y=element_blank(), #去除网格
        panel.border=element_blank(), #去除边框
        axis.line.x=element_line(),
        axis.ticks.length.x=unit(-0.15,"cm"),
        axis.text.x.top=element_text(margin=unit(c(0,0,0.6,0),"cm")), 
        axis.text.y=element_text(margin=unit(c(0.5,0.25,0.5,0.5),"cm")),
        axis.ticks.y=element_blank(),
        plot.margin=unit(c(1,2,1,1),"cm"))

#数据集划分
data_mine <- db[which(db$category=="采矿业"),]
data_electricity <- db[which(db$category=="电力、热力、燃气及水生产和供应业"),]
data_house <- db[which(db$category=="房地产业"),]
data_building <- db[which(db$category=="建筑业"),]
data_transport <- db[which(db$category=="交通运输、仓储和邮政业"),]
data_education <- db[which(db$category=="教育"),]
data_finance <- db[which(db$category=="金融业"),]
data_repair <- db[which(db$category=="居民服务、修理和其他服务业"),]
data_reserch <- db[which(db$category=="科学研究和技术服务业"),]
data_agriculture <- db[which(db$category=="农、林、牧、渔业"),]
data_retail <- db[which(db$category=="批发和零售业"),]
data_environment <- db[which(db$category=="水利、环境和公共设施管理业"),]
data_social <- db[which(db$category=="卫生和社会工作"),]
data_culture <- db[which(db$category=="文化、体育和娱乐业"),]
data_it <- db[which(db$category=="信息传输、软件和信息技术服务业"),]
data_make <- db[which(db$category=="制造业"),]
data_hotel <- db[which(db$category=="住宿和餐饮业"),]
data_total <- db[which(db$category=="综合"),]
data_rent <- db[which(db$category=="租赁和商务服务业"),]
data_count <- rbind(data_education,data_repair,data_social,data_hotel,data_total)

#数据标准化
data_mine_proc <- as.data.frame(scale(data_mine[,-c(1,2,240)],
                                      center = TRUE,scale = TRUE))
as.factor(data_mine$FLAG)
data_mine_proc <- cbind(data_mine_proc,data_mine$FLAG)
colnames(data_mine_proc)[238] <- 'FLAG'
write.csv(data_mine_proc,'F:/essay/data/data_mine.csv')

data_electricity_proc <- as.data.frame(scale(data_electricity[,-c(1,2,240)],
                                             center = TRUE, scale = TRUE))
as.factor(data_electricity$FLAG)
data_electricity_proc <- cbind(data_electricity_proc,data_electricity$FLAG)
colnames(data_electricity_proc)[238] <- 'FLAG'
write.csv(data_electricity_proc,'F:/essay/data/data_electricity.csv')

data_house_proc <- as.data.frame(scale(data_house[,-c(1,2,240)],
                                             center = TRUE, scale = TRUE))
as.factor(data_house$FLAG)
data_house_proc <- cbind(data_house_proc,data_house$FLAG)
colnames(data_house_proc)[238] <- 'FLAG'
write.csv(data_house_proc,'F:/essay/data/data_house.csv')

data_building_proc <- as.data.frame(scale(data_building[,-c(1,2,240)],
                                       center = TRUE, scale = TRUE))
as.factor(data_building$FLAG)
data_building_proc <- cbind(data_building_proc,data_building$FLAG)
colnames(data_building_proc)[238] <- 'FLAG'
write.csv(data_building_proc,'F:/essay/data/data_building.csv')

data_transport_proc <- as.data.frame(scale(data_transport[,-c(1,2,240)],
                                          center = TRUE, scale = TRUE))
as.factor(data_transport$FLAG)
data_transport_proc <- cbind(data_transport_proc,data_transport$FLAG)
colnames(data_transport_proc)[238] <- 'FLAG'
write.csv(data_transport_proc,'F:/essay/data/data_transport.csv')

data_finance_proc <- as.data.frame(scale(data_finance[,-c(1,2,240)],
                                           center = TRUE, scale = TRUE))
as.factor(data_finance$FLAG)
data_finance_proc <- cbind(data_finance_proc,data_finance$FLAG)
colnames(data_finance_proc)[238] <- 'FLAG'
write.csv(data_finance_proc,'F:/essay/data/data_finance.csv')

data_reserch_proc <- as.data.frame(scale(data_reserch[,-c(1,2,240)],
                                         center = TRUE, scale = TRUE))
as.factor(data_reserch$FLAG)
data_reserch_proc <- cbind(data_reserch_proc,data_reserch$FLAG)
colnames(data_reserch_proc)[238] <- 'FLAG'
write.csv(data_reserch_proc,'F:/essay/data/data_reserch.csv')

data_agriculture_proc <- as.data.frame(scale(data_agriculture[,-c(1,2,240)],
                                         center = TRUE, scale = TRUE))
as.factor(data_agriculture$FLAG)
data_agriculture_proc <- cbind(data_agriculture_proc,data_agriculture$FLAG)
colnames(data_agriculture_proc)[238] <- 'FLAG'
write.csv(data_agriculture_proc,'F:/essay/data/data_agriculture.csv')

data_retail_proc <- as.data.frame(scale(data_retail[,-c(1,2,240)],
                                             center = TRUE, scale = TRUE))
as.factor(data_retail$FLAG)
data_retail_proc <- cbind(data_retail_proc,data_retail$FLAG)
colnames(data_retail_proc)[238] <- 'FLAG'
write.csv(data_retail_proc,'F:/essay/data/data_retail.csv')

data_environment_proc <- as.data.frame(scale(data_environment[,-c(1,2,240)],
                                        center = TRUE, scale = TRUE))
as.factor(data_environment$FLAG)
data_environment_proc <- cbind(data_environment_proc,data_environment$FLAG)
colnames(data_environment_proc)[238] <- 'FLAG'
write.csv(data_environment_proc,'F:/essay/data/data_environment.csv')

data_culture_proc <- as.data.frame(scale(data_culture[,-c(1,2,240)],
                                             center = TRUE, scale = TRUE))
as.factor(data_culture$FLAG)
data_culture_proc <- cbind(data_culture_proc,data_culture$FLAG)
colnames(data_culture_proc)[238] <- 'FLAG'
write.csv(data_culture_proc,'F:/essay/data/data_culture.csv')

data_it_proc <- as.data.frame(scale(data_it[,-c(1,2,240)],
                                         center = TRUE, scale = TRUE))
as.factor(data_it$FLAG)
data_it_proc <- cbind(data_it_proc,data_it$FLAG)
colnames(data_it_proc)[238] <- 'FLAG'
write.csv(data_it_proc,'F:/essay/data/data_it.csv')

data_make_proc <- as.data.frame(scale(data_make[,-c(1,2,240)],
                                      center = TRUE, scale = TRUE))
as.factor(data_make$FLAG)
data_make_proc <- cbind(data_make_proc,data_make$FLAG)
colnames(data_make_proc)[238] <- 'FLAG'
write.csv(data_make_proc,'F:/essay/data/data_make.csv')

data_count_proc <- as.data.frame(scale(data_count[,-c(1,2,240)],
                                      center = TRUE, scale = TRUE))
as.factor(data_count$FLAG)
data_count_proc <- cbind(data_count_proc,data_count$FLAG)
colnames(data_count_proc)[238] <- 'FLAG'
write.csv(data_count_proc,'F:/essay/data/data_count.csv')
