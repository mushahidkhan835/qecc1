require(tidyverse)
# dat <- read_csv('stats_gates_storage_optimization_level_zero.csv')
# dat <- read_csv('stats_gates_storage_optimization_level_two_three.csv')
dat <- read_csv('stats_gates_storage.csv')


# make sure that avm.u1, avm.u2, jplf.u1 are not there
dat <- dat %>%
  replace(is.na(.), 0) %>%
  mutate(`gates.ep-pqm_avm.u` = (`gates.ep-pqm.u1` + `gates.ep-pqm.u2` + `gates.ep-pqm.u3`) - (`gates.avm.u2` + `gates.avm.u3`),
         `gates.jplf_avm.u` = (`gates.jplf.u2` + `gates.jplf.u3`) - (`gates.avm.u2` + `gates.avm.u3`),
         `gates.ep-pqm_avm.cx` = (`gates.ep-pqm.cx` - `gates.avm.cx`),
         `gates.jplf_avm.cx` = (`gates.jplf.cx` - `gates.avm.cx`),
         `gates.ep-pqm_avm.u.rel` = `gates.ep-pqm_avm.u` / (`gates.ep-pqm.u1` + `gates.ep-pqm.u2` + `gates.ep-pqm.u3`),
         `gates.ep-pqm_avm.cx.rel` = `gates.ep-pqm_avm.cx` / (`gates.ep-pqm.cx`)
         )

# list where jplf wins over avm
jplf_wins <- dat %>% filter(`gates.jplf_avm.u` < 0) %>% count()
if (jplf_wins > 0) {
  print("u: jplf sometimes wins")
}else{
  print("u: avm sometimes wins")
}

# list where jplf wins over avm
jplf_wins <- dat %>% filter(`gates.jplf_avm.cx` < 0) %>% count()
if (jplf_wins > 0) {
  print("cx: jplf sometimes wins")
}else{
  print("cx: avm sometimes wins")
}


dat.summary <- dat %>%
  group_by(string_length, db_size) %>%
  summarise(
    `gates.ep-pqm_avm.u.min` = min(`gates.ep-pqm_avm.u`),
    `gates.ep-pqm_avm.u.mean` = mean(`gates.ep-pqm_avm.u`),
    `gates.ep-pqm_avm.u.median` = median(`gates.ep-pqm_avm.u`),
    `gates.ep-pqm_avm.u.max` = max(`gates.ep-pqm_avm.u`),

    `gates.ep-pqm_avm.cx.min` = min(`gates.ep-pqm_avm.cx`),
    `gates.ep-pqm_avm.cx.mean` = mean(`gates.ep-pqm_avm.cx`),
    `gates.ep-pqm_avm.cx.median` = median(`gates.ep-pqm_avm.cx`),
    `gates.ep-pqm_avm.cx.max` = max(`gates.ep-pqm_avm.cx`),

    `gates.ep-pqm_avm.u.rel.min` = min(`gates.ep-pqm_avm.u.rel`),
    `gates.ep-pqm_avm.u.rel.mean` = mean(`gates.ep-pqm_avm.u.rel`),
    `gates.ep-pqm_avm.u.rel.median` = median(`gates.ep-pqm_avm.u.rel`),
    `gates.ep-pqm_avm.u.rel.max` = max(`gates.ep-pqm_avm.u.rel`),

    `gates.ep-pqm_avm.cx.rel.min` = min(`gates.ep-pqm_avm.cx.rel`),
    `gates.ep-pqm_avm.cx.rel.mean` = mean(`gates.ep-pqm_avm.cx.rel`),
    `gates.ep-pqm_avm.cx.rel.median` = median(`gates.ep-pqm_avm.cx.rel`),
    `gates.ep-pqm_avm.cx.rel.max` = max(`gates.ep-pqm_avm.cx.rel`),

  )


# Heatmap for cx
ggplot(dat.summary, aes(string_length, db_size, fill= `gates.ep-pqm_avm.cx.median`)) +
  geom_tile() +
  xlab("String length (n)") +
  ylab("Database size (r)") +
  labs(fill = "Gate count") +
  ggtitle("Median savings of cx gates (EP-PQM - Statevector)")

ggplot(dat.summary, aes(string_length, db_size, fill= `gates.ep-pqm_avm.u.median`)) +
  geom_tile() +
  xlab("String length (n)") +
  ylab("Database size (r)") +
  labs(fill = "Gate count") +
  ggtitle("Median savings of u* gates (EP-PQM - Statevector)")


ggplot(dat.summary, aes(string_length, db_size, fill= `gates.ep-pqm_avm.cx.rel.min`)) +
  geom_tile() +
  xlab("String length (n)") +
  ylab("Database size (r)") +
  labs(fill = "Gate fraction") +
  ggtitle("Min savings of cx gates (EP-PQM - Statevector)/EP-PQM")


ggplot(dat.summary, aes(string_length, db_size, fill= `gates.ep-pqm_avm.u.rel.median`)) +
  geom_tile() +
  xlab("String length (n)") +
  ylab("Database size (r)") +
  labs(fill = "Gate fraction") +
  ggtitle("Min savings of u* gates (EP-PQM - Statevector)/EP-PQM")
