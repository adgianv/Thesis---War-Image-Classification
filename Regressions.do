* Set wd
cd "C:\Users\gatla\OneDrive\BSE\Thesis\git_repo\Thesis---War-Image-Classification\data"

*Load monthly csv
import delimited "monthly_panel.csv", clear



* Panel regs - channel fe and clustered standard errors
encode channel, generate(channel_numeric)
generate numdate = date(year_month, "YMD")
gen Gaza_dummy = numdate > date("2023-09-01", "YMD")

xtset channel_numeric numdate

*WHY ARE THE BELOW TWO DIFFERENT???
xtreg coverage fatalities, fe vce(cluster channel_numeric)
xtreg coverage fatalities i.channel_numeric, vce(cluster channel_numeric)


* Regs to analyse

* Coverage
xtreg coverage fatalities, fe vce(cluster channel_numeric)
xtreg coverage launched, fe vce(cluster channel_numeric)


* Checking coverage and war_share_out_of_total show the same trend
xtreg war_share_out_of_total fatalities, fe vce(cluster channel_numeric)
xtreg war_share_out_of_total launched, fe vce(cluster channel_numeric)


* Share of war images
xtreg war_images fatalities, fe vce(cluster channel_numeric)
xtreg war_images launched, fe vce(cluster channel_numeric)


*Gaza dummy
xtreg coverage Gaza_dummy, fe vce(cluster channel_numeric)
xtreg war_images Gaza_dummy, fe vce(cluster channel_numeric)




* Same as above but with weekly aggregation

import delimited "weekly_panel.csv", clear

* Drop final week as not complete data for all channels
drop if week == "2024-04-29"


encode channel, generate(channel_numeric)
generate numdate = date(week, "YMD")
gen Gaza_dummy = numdate > date("2023-10-07", "YMD")

xtset channel_numeric numdate

* Regs to analyse

* Coverage
xtreg coverage fatalities, fe vce(cluster channel_numeric)
xtreg coverage launched, fe vce(cluster channel_numeric)


* Share of war images
xtreg war_images fatalities, fe vce(cluster channel_numeric)
xtreg war_images launched, fe vce(cluster channel_numeric)


* Gaza dummy
xtreg coverage Gaza_dummy, fe vce(cluster channel_numeric)
xtreg war_images Gaza_dummy, fe vce(cluster channel_numeric)


* Find less significance at the weekly aggregation level - but generally no evidence that the coverage or share of war images are driven by events on the ground









