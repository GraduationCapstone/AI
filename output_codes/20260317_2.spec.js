const { test, expect } = require('@playwright/test');

test.describe('Travel Introduction Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('https://danimo1.github.io/travel_introduction_page/');
  });

  test('should display the page title correctly', async ({ page }) => {
    const pageTitle = await page.title();
    expect(pageTitle).toBe('Travel Introduction');
  });

  test('should display the hero section correctly', async ({ page }) => {
    const heroTitle = await page.textContent('h1');
    expect(heroTitle).toBe('Explore the World');

    const heroDescription = await page.textContent('p.hero-description');
    expect(heroDescription).toBe('Discover the beauty and culture of different destinations around the globe.');
  });

  test('should display the image gallery correctly', async ({ page }) => {
    const images = await page.$$eval('img', imgs => imgs.map(img => img.src));
    expect(images).toContain('img/paris-stake.jpg');
    expect(images).toContain('img/london-bigben.JPEG');
    expect(images).toContain('img/brussel-chocolate.jpg');
  });

  test('should navigate to the Paris page correctly', async ({ page }) => {
    await page.click('a[href="paris.html"]');
    await expect(page).toHaveURL(/.*paris.html/);
  });

  test('should navigate to the London page correctly', async ({ page }) => {
    await page.click('a[href="london.html"]');
    await expect(page).toHaveURL(/.*london.html/);
  });

  test('should navigate to the Brussels page correctly', async ({ page }) => {
    await page.click('a[href="brussels.html"]');
    await expect(page).toHaveURL(/.*brussels.html/);
  });
});