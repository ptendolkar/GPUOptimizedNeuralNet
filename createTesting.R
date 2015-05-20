d <- matrix(runif(50)*1.2-.1, ncol=2)
write.table(d, "testing", append=FALSE, quote=FALSE, sep=",", row.names=FALSE, col.names=FALSE)
