---
layout: page
permalink: /publications/
title: Publications
description: Please refer to my [Google Scholar](https://scholar.google.com/citations?user=ZlBEHxwAAAAJ) for a complete publication list.
years: [2026, 2025, 2024, 2023, 2022]
nav: true
nav_order: 2
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>

