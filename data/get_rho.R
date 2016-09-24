
library(ggplot2)

da<-read.table('norvwords_withfreq.txt', header=T)
da<-data.frame(da, seq(nrow(da)))
names(da)[3] <- "k"

# anything above the 75th percentile gets fixed
# why 75? I don't know.
# people should know at least the top 25% most common words (in this limited list)
perc_75<-quantile(da$freq, 0.75)

da[da$freq > perc_75,]$freq <- perc_75

# normalise freqs
total_occurence<-sum(da$freq)
frac<-da$freq/total_occurence
da<-data.frame(da, frac)

# now we need to shift and scale so that the top values are at 1, and the bottom at 0
# but first, log-scaled it
logfreq <- log10(da$freq)
logfreq <- logfreq - min(logfreq)
logfreq <- logfreq/max(logfreq)

da<-data.frame(da, logfreq)
da[, 3] <- NULL

names(da)[4] <- 'rho'
write.table(da, file='norvwords_withfreq_withrho.txt', quote=F, row.names=F, col.names=T)
