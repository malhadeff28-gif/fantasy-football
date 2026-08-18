# QB-019: do quarterbacks genuinely differ in matchup sensitivity?
#
# Prior work rejected per-QB matchup effects using a split-half reliability
# test (r = 0.065). That is suggestive but informal. The standard test is a
# random-slopes mixed model: let each quarterback have his own slope on
# opponent difficulty, with partial pooling, and inspect the variance
# component. If the random-slope standard deviation is essentially zero, the
# effect does not exist -- and unlike fitting 41 separate slopes, this shrinks
# properly rather than manufacturing noise.

suppressMessages({
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], stringsAsFactors = FALSE)
d$player_id <- factor(d$player_id)

cat("rows:", nrow(d), " quarterbacks:", nlevels(d$player_id), "\n\n")

fixed <- "fantasy_points ~ form_fp_mean + career_fp_mean + form_pass_att +
           form_rush_att + implied_team_total + total_line + spread_abs +
           is_home + indoors + def_adj"

# random intercept only: QBs differ in level
m0 <- lmer(as.formula(paste(fixed, "+ (1 | player_id)")), data = d, REML = FALSE)
# random slope on opponent difficulty: QBs differ in matchup sensitivity
m1 <- lmer(as.formula(paste(fixed, "+ (1 + def_adj | player_id)")), data = d,
           REML = FALSE, control = lmerControl(optimizer = "bobyqa"))
# random slope on the market signal: do QBs differ in market sensitivity?
m2 <- lmer(as.formula(paste(fixed, "+ (1 + implied_team_total | player_id)")),
           data = d, REML = FALSE, control = lmerControl(optimizer = "bobyqa"))

cat("=== VARIANCE COMPONENTS ===\n")
cat("\n-- m1: random slope on opponent difficulty --\n")
print(VarCorr(m1))
cat("\n-- m2: random slope on implied team total --\n")
print(VarCorr(m2))

cat("\n=== LIKELIHOOD RATIO TESTS (does the random slope earn its place?) ===\n")
cat("\n-- opponent difficulty --\n")
print(anova(m0, m1))
cat("\n-- implied team total --\n")
print(anova(m0, m2))

cat("\n=== RESIDUAL SD (scale for reading the above) ===\n")
cat("residual SD:", sigma(m1), "fantasy points\n")

cat("\n=== INTERPRETATION SCALE ===\n")
cat("singularity -- a boundary fit means the data cannot identify
")
cat("separate slope variance at all, which is itself the answer
")
cat("m1 singular:", isSingular(m1), " m2 singular:", isSingular(m2), "

")

vc <- as.data.frame(VarCorr(m1))
slope_sd <- vc$sdcor[vc$var1 == "def_adj" & is.na(vc$var2)][1]
cat("random-slope SD on def_adj:", slope_sd, "\n")
cat("def_adj observed SD in data:", sd(d$def_adj, na.rm = TRUE), "\n")
cat("=> a 1-SD swing in opponent difficulty moves the typical QB by",
    round(slope_sd * sd(d$def_adj, na.rm = TRUE), 4),
    "extra fantasy points relative to the league-average response\n")
