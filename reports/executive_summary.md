# Executive Summary - Marketplace Profitability

**Headline:** 40,000 orders generated INR 204.87M net revenue and INR 14.9M profit
(**7.27% margin**). 18.9% of orders lose money, and 77.3% of those
involve either a discount above 20% or a customer return.

## What drives profit
1. **Category mix.** Electronics contributes the most profit (INR 4.0M); Beauty & Personal Care earns the best margin
   (30.8%) while Appliances earns the thinnest (2.6%).
2. **Discount depth.** Loss-making rate climbs from 9.0% at 0-10% discount to
   87.7% above 40%. Orders discounted over 20% *destroy* INR
   2.8M of profit.
3. **Seasonality.** Festive months (Jul, Oct, Nov) average 18.8% discount vs 12.7% normally, cutting
   margin from 9.83% to **2.34%** despite record revenue.

## Where we lose money
- **Returns:** 6.8% of orders are returned, costing INR 6.38M; Fashion is worst at 9.5%.
- **Pricing leakage:** INR 3.01M lost on orders that were *kept* by customers but sold below cost-to-serve.
- **Logistics:** Same-day orders earn INR 266 per order vs INR 405 for standard;
  North-East earns INR 292 vs INR 402 in the North.

## Recommendations
1. **Cap discounts at 25%** (affects 12.0% of orders). Even assuming 15% of affected orders are lost,
   profit rises by **INR 2.01M (+13.5%)**.
2. **Use the loss-risk model at pricing time** to block or reprice orders flagged as likely loss-makers.
3. **Attack returns** in Fashion (size guidance, quality checks) and add a return-risk surcharge for repeat returners.

## Model performance
The best model (Gradient Boosting) predicts whether an order will lose money with **94.1% accuracy**
on a chronological hold-out (2025-09-24 to 2025-12-31), versus a 77.0% majority-class baseline;
precision 99.5%, recall 74.8%, AUC 0.901.
Of 1839 true loss-making orders it caught 1375; 99.6% of the misses were
caused by customer returns, which cannot be known at order time.

## Limitations
Data is simulated, so absolute figures illustrate the method, not Amazon's real economics. Discount scenarios
assume 15% volume loss on repriced orders; real elasticity should be measured with an A/B test.
